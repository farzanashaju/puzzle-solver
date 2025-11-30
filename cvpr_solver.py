import numpy as np
import cv2
import heapq
import os
import csv
import time
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# --- CONFIGURATION ---
GRID_ROWS = 8  
GRID_COLS = 8
# ---------------------

class JigsawPiece:
    def __init__(self, piece_id, image_data, true_pos):
        self.id = piece_id
        self.image = image_data
        self.true_pos = true_pos # (row, col)
        self.h, self.w = image_data.shape[:2]
        self.lab_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Extract boundaries
        self.top = self.lab_image[0, :, :]
        self.bottom = self.lab_image[-1, :, :]
        self.left = self.lab_image[:, 0, :]
        self.right = self.lab_image[:, -1, :]
        
        # Extract inner boundaries (for gradient calculation)
        self.top_inner = self.lab_image[1, :, :]
        self.bottom_inner = self.lab_image[-2, :, :]
        self.left_inner = self.lab_image[:, 1, :]
        self.right_inner = self.lab_image[:, -2, :]

class PaikinTalSolver:
    def __init__(self, piece_h, piece_w, grid_rows, grid_cols):
        self.pieces = []
        self.piece_h = piece_h
        self.piece_w = piece_w
        self.num_pieces = 0
        self.dissimilarity = None
        self.compatibility = None
        self.best_buddies = set()
        self.relations = [0, 1, 2, 3] # 0:Right, 1:Down, 2:Left, 3:Up
        
        # Dimensions constraint
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def add_piece(self, piece_obj):
        self.pieces.append(piece_obj)
        self.num_pieces += 1

    def calculate_dissimilarity(self):
        N = self.num_pieces
        self.dissimilarity = np.full((N, N, 4), np.inf)

        # Vectorized or simple loop (using simple loop for clarity as per previous code)
        # Note: In a production loop, we might suppress the tqdm here to avoid clutter
        for i in range(N):
            for j in range(N):
                if i == j: continue
                
                # Right
                pred_right = 2 * self.pieces[i].right - self.pieces[i].right_inner
                diff_right = pred_right - self.pieces[j].left
                d_right = np.sum(np.abs(diff_right))
                self.dissimilarity[i, j, 0] = d_right
                self.dissimilarity[j, i, 2] = d_right 

                # Down
                pred_down = 2 * self.pieces[i].bottom - self.pieces[i].bottom_inner
                diff_down = pred_down - self.pieces[j].top
                d_down = np.sum(np.abs(diff_down))
                self.dissimilarity[i, j, 1] = d_down
                self.dissimilarity[j, i, 3] = d_down

    def calculate_compatibility(self):
        N = self.num_pieces
        self.compatibility = np.zeros((N, N, 4))
        
        for i in range(N):
            for r in self.relations:
                dists = self.dissimilarity[i, :, r].copy()
                dists[i] = np.inf 
                sorted_indices = np.argsort(dists)
                second_best_val = dists[sorted_indices[1]]
                if second_best_val == 0: second_best_val = 1e-6

                for j in range(N):
                    if i == j: continue
                    d_val = self.dissimilarity[i, j, r]
                    comp = 1 - (d_val / second_best_val)
                    self.compatibility[i, j, r] = max(0, comp)

    def find_best_buddies(self):
        self.best_buddies = set()
        N = self.num_pieces
        for i in range(N):
            for r in self.relations:
                best_j = np.argmin(self.dissimilarity[i, :, r])
                inv_r = (r + 2) % 4
                best_i_candidate = np.argmin(self.dissimilarity[best_j, :, inv_r])
                if best_i_candidate == i:
                    self.best_buddies.add((i, best_j, r))

    def get_mutual_compatibility(self, i, j, r):
        inv_r = (r + 2) % 4
        return (self.compatibility[i, j, r] + self.compatibility[j, i, inv_r]) / 2

    def find_first_piece(self, available_pieces):
        best_piece = -1
        max_score = -1
        candidates = list(available_pieces)
        
        for pid in candidates:
            # Check for best buddies in all directions
            buddy_directions = set()
            for (u, v, rel) in self.best_buddies:
                if u == pid: buddy_directions.add(rel)
            
            if len(buddy_directions) < 4:
                continue
            
            score = 0
            for r in range(4):
                buddy = -1
                for (u, v, rel) in self.best_buddies:
                    if u == pid and rel == r:
                        buddy = v
                        break
                if buddy != -1:
                    score += self.get_mutual_compatibility(pid, buddy, r)
            
            if score > max_score:
                max_score = score
                best_piece = pid
        
        if best_piece == -1 and candidates:
             return candidates[0]
        return best_piece

    def _is_valid_placement(self, current_solution, new_pos):
        if not self.grid_cols or not self.grid_rows:
            return True 
        
        xs = [p[0] for p in current_solution.keys()]
        ys = [p[1] for p in current_solution.keys()]
        
        xs.append(new_pos[0])
        ys.append(new_pos[1])
        
        width_span = max(xs) - min(xs) + 1
        height_span = max(ys) - min(ys) + 1
        
        if width_span > self.grid_cols or height_span > self.grid_rows:
            return False
        return True

    def get_open_boundary_slots(self, current_solution):
        open_slots = set()
        for pos in current_solution.keys():
            x, y = pos
            for dx, dy, r in [(1,0,0), (0,1,1), (-1,0,2), (0,-1,3)]:
                neighbor_pos = (x + dx, y + dy)
                if neighbor_pos not in current_solution:
                    if self._is_valid_placement(current_solution, neighbor_pos):
                        open_slots.add((neighbor_pos, r, current_solution[pos]))
        return list(open_slots)

    def _add_candidates_to_pool(self, placed_id, placed_pos, pool, unplaced, current_sol):
        for r in self.relations:
            dx, dy = {0: (1,0), 1: (0,1), 2: (-1,0), 3: (0,-1)}[r]
            target_pos = (placed_pos[0] + dx, placed_pos[1] + dy)
            
            if target_pos in current_sol: continue
            if not self._is_valid_placement(current_sol, target_pos): continue

            found_buddy = False
            for (u, v, rel) in self.best_buddies:
                if u == placed_id and rel == r and v in unplaced:
                    comp = self.get_mutual_compatibility(u, v, r)
                    heapq.heappush(pool, (-comp, placed_id, v, r))
                    found_buddy = True
            
            if not found_buddy:
                best_j = -1
                best_mut_comp = -1.0
                for cand_id in unplaced:
                    comp = self.get_mutual_compatibility(placed_id, cand_id, r)
                    if comp > best_mut_comp:
                        best_mut_comp = comp
                        best_j = cand_id
                
                if best_j != -1:
                    heapq.heappush(pool, (-best_mut_comp, placed_id, best_j, r))

    def solve(self):
        self.calculate_dissimilarity()
        self.calculate_compatibility()
        self.find_best_buddies()
        
        unplaced_pieces = set(range(self.num_pieces))
        current_solution = {} 
        candidate_pool = [] 
        
        # 1. Place Seed
        first_piece = self.find_first_piece(unplaced_pieces)
        current_solution[(0, 0)] = first_piece
        unplaced_pieces.remove(first_piece)
        self._add_candidates_to_pool(first_piece, (0,0), candidate_pool, unplaced_pieces, current_solution)
        
        while unplaced_pieces:
            placed_flag = False
            
            # --- PHASE 1: Greedy using Heap ---
            while candidate_pool:
                score, src_id, best_match_id, rel = heapq.heappop(candidate_pool)
                
                if best_match_id not in unplaced_pieces: continue
                
                src_pos = next((pos for pos, pid in current_solution.items() if pid == src_id), None)
                if src_pos is None: continue 

                dx, dy = {0: (1,0), 1: (0,1), 2: (-1,0), 3: (0,-1)}[rel]
                new_pos = (src_pos[0] + dx, src_pos[1] + dy)
                
                if new_pos in current_solution: continue
                if not self._is_valid_placement(current_solution, new_pos): continue

                current_solution[new_pos] = best_match_id
                unplaced_pieces.remove(best_match_id)
                self._add_candidates_to_pool(best_match_id, new_pos, candidate_pool, unplaced_pieces, current_solution)
                placed_flag = True
                break
            
            if placed_flag: continue

            # --- PHASE 2: Forced Placement ---
            open_slots = self.get_open_boundary_slots(current_solution)
            if not open_slots: break 

            best_global_score = -1.0
            best_move = None 
            
            for (target_pos, rel, placed_id) in open_slots:
                for cand_id in unplaced_pieces:
                    comp = self.get_mutual_compatibility(placed_id, cand_id, rel)
                    if comp > best_global_score:
                        best_global_score = comp
                        best_move = (cand_id, target_pos, rel, placed_id)
            
            if best_move:
                cand_id, target_pos, rel, src_id = best_move
                current_solution[target_pos] = cand_id
                unplaced_pieces.remove(cand_id)
                self._add_candidates_to_pool(cand_id, target_pos, candidate_pool, unplaced_pieces, current_solution)
            else:
                break

        return current_solution

    def reconstruct_image(self, solution_dict):
        if not solution_dict: return None
        coords = list(solution_dict.keys())
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        
        min_x, min_y = min(xs), min(ys)
        
        full_h = self.grid_rows * self.piece_h
        full_w = self.grid_cols * self.piece_w
        
        canvas = np.zeros((full_h, full_w, 3), dtype=np.uint8)
        
        # We need to map the relative grid to the 0-indexed grid for reconstruction
        for pos, pid in solution_dict.items():
            # Adjust coordinates so the top-left-most piece is at (0,0)
            adj_x = pos[0] - min_x
            adj_y = pos[1] - min_y
            
            px = adj_x * self.piece_w
            py = adj_y * self.piece_h
            
            if px < full_w and py < full_h:
                canvas[py:py+self.piece_h, px:px+self.piece_w] = self.pieces[pid].image
            
        return canvas

def calculate_metrics(solution_dict, solver, original_image):
    """
    Calculates Direct Accuracy, Neighbor Accuracy, and SSIM.
    """
    if not solution_dict:
        return {'direct_accuracy': 0, 'neighbor_accuracy': 0, 'ssim': 0}

    # 1. Normalize Coordinates (Shift so top-left is 0,0)
    coords = list(solution_dict.keys())
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    min_x, min_y = min(xs), min(ys)
    
    normalized_solution = {}
    for pos, pid in solution_dict.items():
        new_x = pos[0] - min_x
        new_y = pos[1] - min_y
        normalized_solution[(new_x, new_y)] = solver.pieces[pid]

    # 2. Direct Accuracy
    total_pieces = solver.grid_rows * solver.grid_cols
    correct_positions = 0
    
    for c in range(solver.grid_cols):
        for r in range(solver.grid_rows):
            if (c, r) in normalized_solution:
                piece = normalized_solution[(c, r)]
                # piece.true_pos is (row, col)
                if piece.true_pos == (r, c):
                    correct_positions += 1
    
    direct_acc = correct_positions / total_pieces

    # 3. Neighbor Accuracy
    total_neighbors = 0
    correct_neighbors = 0
    
    for c in range(solver.grid_cols):
        for r in range(solver.grid_rows):
            if (c, r) in normalized_solution:
                current_piece = normalized_solution[(c, r)]
                
                # Check Right
                if (c + 1, r) in normalized_solution:
                    right_piece = normalized_solution[(c + 1, r)]
                    total_neighbors += 1
                    # Check if they are truly neighbors horizontally
                    if current_piece.true_pos[0] == right_piece.true_pos[0] and \
                       current_piece.true_pos[1] + 1 == right_piece.true_pos[1]:
                        correct_neighbors += 1
                
                # Check Down
                if (c, r + 1) in normalized_solution:
                    down_piece = normalized_solution[(c, r + 1)]
                    total_neighbors += 1
                    # Check if they are truly neighbors vertically
                    if current_piece.true_pos[0] + 1 == down_piece.true_pos[0] and \
                       current_piece.true_pos[1] == down_piece.true_pos[1]:
                        correct_neighbors += 1
                        
    neighbor_acc = correct_neighbors / total_neighbors if total_neighbors > 0 else 0

    # 4. SSIM
    reconstructed = solver.reconstruct_image(solution_dict)
    
    # Original image might need to be resized to match reconstructed exactly 
    # (though our logic ensured they are same size, safety check)
    if reconstructed.shape != original_image.shape:
        original_image = cv2.resize(original_image, (reconstructed.shape[1], reconstructed.shape[0]))
        
    gray_orig = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_recon = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
    ssim_val = ssim(gray_orig, gray_recon, data_range=255)

    return {
        'direct_accuracy': direct_acc,
        'neighbor_accuracy': neighbor_acc,
        'ssim': ssim_val
    }

def main():
    # suppress numpy warnings
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # create output directory
    output_dir = Path('cvpr_output')
    output_dir.mkdir(exist_ok=True)

    # find images in dataset directory
    dataset_dir = Path('dataset')
    
    # get all jpg and png images
    image_files = sorted(list(dataset_dir.glob('*.jpg')) + list(dataset_dir.glob('*.png')))
    
    if not image_files:
        print(f"No images found in {dataset_dir}. Please create folder and add images.")
        return
    
    # prepare CSV file for detailed metrics
    detailed_csv_path = output_dir / 'detailed_metrics.csv'
    detailed_csv_file = open(detailed_csv_path, 'w', newline='')
    detailed_csv_writer = csv.writer(detailed_csv_file)
    
    # write header
    detailed_csv_writer.writerow([
        'image_name', 'strategy', 'direct_accuracy', 
        'neighbor_accuracy', 'ssim'
    ])
    
    # dictionary to accumulate metrics for aggregation
    strategy_metrics = {}
    strategy_name = "PaikinTal_Constraint" # Since this code runs one main algo

    # process each image
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
        print("-" * 60)
        
        try:
            # Load Image
            img = cv2.imread(str(image_path))
            if img is None: continue
            
            h_orig, w_orig = img.shape[:2]
            
            # 1. Resize/Crop logic to fit grid perfectly
            piece_h = h_orig // GRID_ROWS
            piece_w = w_orig // GRID_COLS
            target_h = piece_h * GRID_ROWS
            target_w = piece_w * GRID_COLS
            
            # Resize image to exact multiple of grid
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
            # 2. Create Pieces and Solver
            pieces_img = []
            solver = PaikinTalSolver(piece_h, piece_w, GRID_ROWS, GRID_COLS)
            
            piece_id = 0
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    y = r * piece_h
                    x = c * piece_w
                    p_img = img[y : y + piece_h, x : x + piece_w]
                    
                    # Store piece with Ground Truth info
                    piece_obj = JigsawPiece(piece_id, p_img, true_pos=(r, c))
                    pieces_img.append(piece_obj)
                    piece_id += 1

            # Shuffle
            np.random.seed(42 + idx) # Seed changes per image for variance but is reproducible
            np.random.shuffle(pieces_img)
            
            # Add to solver (note: IDs in solver will be 0..N based on add order, 
            # but piece_obj.id retains the original ID or we track via obj)
            for p in pieces_img:
                solver.add_piece(p)
            
            # 3. Solve
            start_time = time.time()
            solution = solver.solve()
            elapsed = time.time() - start_time
            print(f"Solved in {elapsed:.2f}s")
            
            # 4. Evaluate
            metrics = calculate_metrics(solution, solver, img)
            
            print(f"Direct Accuracy: {metrics['direct_accuracy']:.2%}")
            print(f"Neighbor Accuracy: {metrics['neighbor_accuracy']:.2%}")
            print(f"SSIM: {metrics['ssim']:.4f}")
            
            # 5. Save Solution Image
            result_img = solver.reconstruct_image(solution)
            if result_img is not None:
                res_path = output_dir / f'result_{image_path.stem}.jpg'
                cv2.imwrite(str(res_path), result_img)
            
            # 6. Log Metrics
            detailed_csv_writer.writerow([
                image_path.name,
                strategy_name,
                f"{metrics['direct_accuracy']:.4f}",
                f"{metrics['neighbor_accuracy']:.4f}",
                f"{metrics['ssim']:.4f}"
            ])
            
            if strategy_name not in strategy_metrics:
                strategy_metrics[strategy_name] = {
                    'direct_accuracy': [],
                    'neighbor_accuracy': [],
                    'ssim': []
                }
            strategy_metrics[strategy_name]['direct_accuracy'].append(metrics['direct_accuracy'])
            strategy_metrics[strategy_name]['neighbor_accuracy'].append(metrics['neighbor_accuracy'])
            strategy_metrics[strategy_name]['ssim'].append(metrics['ssim'])

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # close detailed CSV file
    detailed_csv_file.close()
    print(f"\n{'='*60}")
    print(f"Detailed metrics saved to: {detailed_csv_path}")
    
    # create aggregated metrics CSV
    aggregated_csv_path = output_dir / 'aggregated_metrics.csv'
    with open(aggregated_csv_path, 'w', newline='') as agg_csv_file:
        agg_csv_writer = csv.writer(agg_csv_file)
        
        # write header
        agg_csv_writer.writerow([
            'strategy', 'avg_direct_accuracy', 'avg_neighbor_accuracy', 
            'avg_ssim', 'std_direct_accuracy', 'std_neighbor_accuracy', 'std_ssim'
        ])
        
        # write aggregated metrics
        for s_name in sorted(strategy_metrics.keys()):
            m = strategy_metrics[s_name]
            agg_csv_writer.writerow([
                s_name,
                f"{np.mean(m['direct_accuracy']):.4f}",
                f"{np.mean(m['neighbor_accuracy']):.4f}",
                f"{np.mean(m['ssim']):.4f}",
                f"{np.std(m['direct_accuracy']):.4f}",
                f"{np.std(m['neighbor_accuracy']):.4f}",
                f"{np.std(m['ssim']):.4f}"
            ])
    
    print(f"Aggregated metrics saved to: {aggregated_csv_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()