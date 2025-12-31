import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import argparse
import re
from collections import defaultdict
import math
import gc
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

# --- 高速化のための正規表現プリコンパイル ---
RE_LORA_TE1 = re.compile(r"^lora_te1_")
RE_LORA_TE2 = re.compile(r"^lora_te2_")
RE_LORA_TE = re.compile(r"^lora_te_")
RE_LORA_UNET = re.compile(r"^lora_unet_")
RE_TEXT_ENC_2 = re.compile(r"^text_encoder_2\.")
RE_TEXT_ENC = re.compile(r"^text_encoder\.")
RE_PROCESSOR = re.compile(r"^processor\.")
RE_TE_TEXT_MODEL = re.compile(r"^te_text_model\.")
RE_LORA_UP_DOT = re.compile(r"lora\.up")
RE_LORA_DOWN_DOT = re.compile(r"lora\.down")
RE_MULTI_DOT = re.compile(r'\.+')

def unify_key(key):
    if key.startswith("lora_te1_"): key = RE_LORA_TE1.sub("te1.", key)
    elif key.startswith("lora_te2_"): key = RE_LORA_TE2.sub("te2.", key)
    elif key.startswith("lora_te_"): key = RE_LORA_TE.sub("te1.", key) 
    elif key.startswith("lora_unet_"): key = RE_LORA_UNET.sub("unet.", key)
    
    if key.startswith("text_encoder_2."): key = RE_TEXT_ENC_2.sub("te2.", key)
    elif key.startswith("text_encoder."): key = RE_TEXT_ENC.sub("te1.", key)
    
    if key.startswith("processor."): key = RE_PROCESSOR.sub("", key)
    if key.startswith("te_text_model."): key = RE_TE_TEXT_MODEL.sub("text_model.", key)

    unified = key.replace("_", ".")
    
    if "lora.up" in unified: unified = RE_LORA_UP_DOT.sub("lora_up", unified)
    if "lora.down" in unified: unified = RE_LORA_DOWN_DOT.sub("lora_down", unified)
    if "weight" not in unified: unified += ".weight"
    
    unified = RE_MULTI_DOT.sub('.', unified)
    return unified

def find_safetensors_recursive(directory):
    matches = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.safetensors'):
                matches.append(os.path.join(root, filename))
    return matches

class StreamingCovariance:
    def __init__(self, device="cuda", dim=None):
        self.device = device
        self.n = 0
        self.dim = dim
        self.sum_x = None
        self.sum_xtx = None

    def _init_buffers(self, dim):
        self.dim = dim
        # 【重要】蓄積バッファは float32 を維持して精度崩壊を防ぐ
        # ※ もしここも bf16 にすれば VRAM はさらに半減しますが、
        #    数千ファイルの微細な差分が潰れてしまうため推奨しません。
        self.sum_x = torch.zeros(self.dim, device=self.device, dtype=torch.float32)
        self.sum_xtx = torch.zeros((self.dim, self.dim), device=self.device, dtype=torch.float32)

    def update_batch(self, x_batch):
        if x_batch.numel() == 0: return
        if self.sum_x is None: 
            self._init_buffers(x_batch.shape[1])
        
        if x_batch.shape[1] != self.dim: 
            return

        # 1. 転送: ここまでは bf16 で高速・軽量に持ってくる
        x_bf16 = x_batch.to(self.device, dtype=torch.bfloat16, non_blocking=True)
        self.n += x_bf16.shape[0]

        # 2. キャスト: 計算直前に float32 に戻す
        # バッチサイズ分(200x23040程度)の一時メモリしか食わないので、VRAMは安全です
        x_float = x_bf16.float()

        # 3. 計算 & 蓄積: float32 同士ならエラーは起きない
        self.sum_x.add_(x_float.sum(dim=0))
        self.sum_xtx.addmm_(x_float.T, x_float)

    def fit_pca(self, k):
        if self.n <= 1 or self.sum_x is None: return None, None
        
        try:
            # 平均の計算
            mean = self.sum_x / self.n
            
            # 共分散行列の計算 (分散公式: E[XX^T] - E[X]E[X]^T)
            # sum_xtx から (sum_x * sum_x.T) / n を引く
            self.sum_xtx.addmm_(self.sum_x.unsqueeze(1), self.sum_x.unsqueeze(0), alpha=-1.0/self.n)
            
            # 不要メモリ解放
            del self.sum_x
            self.sum_x = None
            
            # 【削除】ここでエラーになっていた対称化処理 (add_, mul_) は、
            # linalg.eigh が下三角しか見ないため、実は削除しても問題ありません。
            # これにより「自分自身への転置加算エラー」を回避しつつ、メモリも節約できます。
            
            # 固有値分解 (float32)
            # UPLO='L' (デフォルト) なので、下半分だけ見て計算してくれます
            L, V = torch.linalg.eigh(self.sum_xtx)
            
            # メモリ解放
            del self.sum_xtx
            self.sum_xtx = None
            
            idx = torch.argsort(L, descending=True)
            V = V[:, idx]
            
            k = min(k, V.shape[1])
            return mean.float().cpu(), V[:, :k].float().cpu()

        except torch.cuda.OutOfMemoryError:
            print(f" VRAM limit reached. Falling back...")
            torch.cuda.empty_cache()
            return None, None
        except Exception as e:
            print(f"PCA Error: {e}")
            return None, None

def analyze_file_metadata(f_path):
    local_map = []
    local_dims = {}
    try:
        with safe_open(f_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            for k in keys:
                uk = unify_key(k)
                if "alpha" in uk or "scale" in uk: continue
                
                slice_info = f.get_slice(k)
                shape = slice_info.get_shape()
                
                dim_size = 0
                if len(shape) == 1:
                    dim_size = shape[0]
                else:
                    if "lora_up" in uk or "lora.up" in uk: 
                        dim_size = shape[0] 
                    else:
                        dim_size = 1
                        for s in shape[1:]: dim_size *= s
                            
                local_map.append((uk, k))
                local_dims[uk] = dim_size
    except Exception:
        pass
    return f_path, local_map, local_dims

def load_file_content_batch(f_path, required_keys):
    loaded_data = {}
    try:
        with safe_open(f_path, framework="pt", device="cpu") as f:
            for u_key, raw_key in required_keys:
                try:
                    tensor = f.get_tensor(raw_key)
                    if tensor.dim() == 1: 
                        tensor = tensor.unsqueeze(0)
                    else:
                        if "lora_up" in u_key or "lora.up" in u_key: 
                            tensor = tensor.transpose(0, 1) 
                        tensor = tensor.flatten(start_dim=1)
                    
                    loaded_data[u_key] = tensor.float()
                except: pass
    except Exception:
        pass
    return loaded_data

def extract_universal_subspace(lora_dir, output_path, num_components=16, device="cuda", vram_fraction=0.9, num_workers=8, chunk_size=200):
    if not torch.cuda.is_available() and device == "cuda": device = "cpu"
    
    # 元のスレッド設定を保存し、読み込みフェーズは1スレッドに制限（コンテキストスイッチ対策）
    original_threads = torch.get_num_threads()
    # もし初期値が1なら、CPUコア数を取得して設定
    if original_threads == 1:
        try:
            original_threads = multiprocessing.cpu_count()
        except:
            original_threads = 4

    torch.set_num_threads(1)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Scanning LoRAs in {lora_dir}...")
    lora_files = find_safetensors_recursive(lora_dir)
    if not lora_files: raise ValueError("No files found.")

    print(f"Indexing {len(lora_files)} files...")
    
    layer_map = defaultdict(list)
    layer_dims = {}
    
    # メタデータ取得フェーズ
    with ThreadPoolExecutor(max_workers=min(num_workers, len(lora_files))) as executor:
        futures = [executor.submit(analyze_file_metadata, f) for f in lora_files]
        for future in tqdm(as_completed(futures), total=len(lora_files), desc="Indexing Keys"):
            f_path, l_map, l_dims = future.result()
            if not l_map: continue
            for uk, raw_key in l_map:
                layer_map[uk].append((f_path, raw_key))
                if uk not in layer_dims:
                    layer_dims[uk] = l_dims[uk]

    # === キーの出現頻度フィルタリング（2ファイル以下は除外） ===
    print("Filtering keys appearing in 2 or fewer files...")
    initial_key_count = len(layer_map)
    
    keys_to_remove = [k for k, v in layer_map.items() if len(v) <= 2]
    
    for k in keys_to_remove:
        del layer_map[k]
        if k in layer_dims:
            del layer_dims[k]
            
    print(f"Filtered {len(keys_to_remove)} rare keys. Remaining unique layers: {len(layer_map)}")
    # =================================

    valid_keys = list(layer_map.keys())
    if not valid_keys: 
        print("No valid keys remaining after filtering.")
        return

    # VRAMに基づくバッチ分割
    if device == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory
        available_mem = total_vram * vram_fraction
        print(f"Available VRAM Target: {available_mem / 1024**3:.2f} GB")
    else:
        available_mem = 32 * 1024**3 

    batches = []
    current_batch = []
    current_mem = 0
    sorted_keys = sorted(valid_keys)
    
    # PCA計算時の一時メモリを考慮し、バッチ制限を少し厳しくする（available_memの80%までで区切る）
    batch_limit = available_mem * 0.8
    
    for key in sorted_keys:
        d = layer_dims.get(key, 0)
        if d == 0: continue
        # 共分散行列の保存に必要なメモリ
        mem_needed = d*d*8 + d*8
        
        if current_mem + mem_needed > batch_limit:
            if current_batch: batches.append(current_batch)
            current_batch = [key]
            current_mem = mem_needed
        else:
            current_batch.append(key)
            current_mem += mem_needed
    if current_batch: batches.append(current_batch)

    print(f"Processing split into {len(batches)} groups.")
    
    universal_state_dict = {}
    
    for b_idx, current_keys in enumerate(batches):
        print(f"\n--- Group {b_idx+1}/{len(batches)} ({len(current_keys)} layers) ---")
        torch.cuda.empty_cache()
        gc.collect()
        
        streamers = {k: StreamingCovariance(device, dim=layer_dims[k]) for k in current_keys}
        current_keys_set = set(current_keys)
        
        files_needed = defaultdict(list)
        for uk in current_keys:
            for f_path, raw_key in layer_map[uk]:
                files_needed[f_path].append((uk, raw_key))
        
        target_file_list = list(files_needed.keys())
        total_files = len(target_file_list)
        
        actual_workers = min(num_workers, total_files)
        if actual_workers < 1: actual_workers = 1
        
        print(f"Processing {total_files} files with {actual_workers} workers...")
        
        accum_buffer = defaultdict(list)
        processed_count = 0
        
        # データの読み込みフェーズ
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            file_iter = iter(target_file_list)
            futures = set()
            
            for _ in range(min(actual_workers + 2, total_files)):
                try:
                    f = next(file_iter)
                    futures.add(executor.submit(load_file_content_batch, f, files_needed[f]))
                except StopIteration: break
            
            with tqdm(total=total_files, desc="Accumulating Stats") as pbar:
                while futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    
                    for fut in done:
                        try:
                            res = fut.result()
                            if res:
                                for u_key, tensor in res.items():
                                    if u_key in current_keys_set:
                                        if tensor.shape[1] == layer_dims[u_key]:
                                            accum_buffer[u_key].append(tensor)
                        except Exception as e:
                            print(f"Error reading file: {e}")

                        processed_count += 1
                        pbar.update(1)

                        try:
                            next_f = next(file_iter)
                            futures.add(executor.submit(load_file_content_batch, next_f, files_needed[next_f]))
                        except StopIteration: pass

                    is_chunk_full = (processed_count > 0 and processed_count % chunk_size == 0)
                    is_last = (len(futures) == 0)
                    
                    if (is_chunk_full or is_last) and accum_buffer:
                        for u_key, tensor_list in accum_buffer.items():
                            if not tensor_list: continue
                            try:
                                batch_tensor = torch.cat(tensor_list, dim=0)
                                streamers[u_key].update_batch(batch_tensor)
                            except Exception as e:
                                print(f"Batch update error {u_key}: {e}")
                        accum_buffer.clear()
                        gc.collect()

        # PCA計算フェーズ: スレッド数を戻して計算能力を確保
        print("Computing PCA...")
        torch.set_num_threads(original_threads) 
        
        for k in tqdm(current_keys, desc="PCA"):
            mean, basis = streamers[k].fit_pca(num_components)
            if mean is not None:
                universal_state_dict[f"{k}.mean"] = mean
                universal_state_dict[f"{k}.basis"] = basis
            del streamers[k]
        
        # 次のグループのためにスレッド数を再び制限（安全のため）
        torch.set_num_threads(1)
        del streamers
        torch.cuda.empty_cache()

    print(f"Saving to {output_path}...")
    save_file(universal_state_dict, output_path)
    print("Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="universal_lora.safetensors")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vram_fraction", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=8) 
    parser.add_argument("--chunk_size", type=int, default=200)
    
    args = parser.parse_args()

    extract_universal_subspace(
        args.lora_dir, 
        args.output, 
        args.rank, 
        args.device, 
        args.vram_fraction,
        args.num_workers,
        args.chunk_size
    )