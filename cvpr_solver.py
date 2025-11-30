import numpy as np
import cv2
import heapq
import os
from tqdm import tqdm

IMAGE_PATH = "dataset/image_3.jpg"
OUTPUT_DIR = "cvpr_output"
PIECE_SIZE = 28

class JigsawPiece:
    def __init__(self, piece_id, image_data):
        self.id = piece_id
        self.image = image_data
        self.size = image_data.shape[0]
        self.lab_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        self.top = self.lab_image[0, :, :]
        self.bottom = self.lab_image[-1, :, :]
        self.left = self.lab_image[:, 0, :]
        self.right = self.lab_image[:, -1, :]
        
        self.top_inner = self.lab_image[1, :, :]
        self.bottom_inner = self.lab_image[-2, :, :]
        self.left_inner = self.lab_image[:, 1, :]
        self.right_inner = self.lab_image[:, -2, :]

class PaikinTalSolver:
    def __init__(self, piece_size, grid_width=None, grid_height=None):
        self.pieces = []
        self.piece_size = piece_size
        self.num_pieces = 0
        self.dissimilarity = None
        self.compatibility = None
        self.best_buddies = set()
        self.relations = [0, 1, 2, 3] # 0:Right, 1:Down, 2:Left, 3:Up
        
        # Dimensions constraint (in number of pieces)
        self.grid_w = grid_width
        self.grid_h = grid_height

    def add_piece(self, image_data):
        piece = JigsawPiece(len(self.pieces), image_data)
        self.pieces.append(piece)
        self.num_pieces += 1

    def calculate_dissimilarity(self):
        print("1/4: Calculating Dissimilarity Matrix...")
        N = self.num_pieces
        self.dissimilarity = np.full((N, N, 4), np.inf)

        for i in tqdm(range(N)):
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
        print("2/4: Calculating Compatibility Matrix...")
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
        print("3/4: Finding Best Buddies...")
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
            
            # If not strictly 4 buddies, we can relax it slightly or prioritize score
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
        """Checks if placing a piece at new_pos violates the known grid dimensions."""
        if not self.grid_w or not self.grid_h:
            return True # No constraints
        
        # Current bounds
        xs = [p[0] for p in current_solution.keys()]
        ys = [p[1] for p in current_solution.keys()]
        
        # Proposed new bounds
        xs.append(new_pos[0])
        ys.append(new_pos[1])
        
        width_span = max(xs) - min(xs) + 1
        height_span = max(ys) - min(ys) + 1
        
        if width_span > self.grid_w or height_span > self.grid_h:
            return False
            
        return True

    def get_open_boundary_slots(self, current_solution):
        open_slots = set()
        for pos in current_solution.keys():
            x, y = pos
            for dx, dy, r in [(1,0,0), (0,1,1), (-1,0,2), (0,-1,3)]:
                neighbor_pos = (x + dx, y + dy)
                if neighbor_pos not in current_solution:
                    # Check dimension constraints before adding to list
                    if self._is_valid_placement(current_solution, neighbor_pos):
                        open_slots.add((neighbor_pos, r, current_solution[pos]))
        return list(open_slots)

    def solve(self):
        self.calculate_dissimilarity()
        self.calculate_compatibility()
        self.find_best_buddies()
        
        unplaced_pieces = set(range(self.num_pieces))
        current_solution = {} 
        candidate_pool = [] 
        
        print(f"4/4: Assembling with strict size constraint: {self.grid_w}x{self.grid_h} pieces...")
        
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
                
                # KEY CHANGE: Check Dimensions
                if not self._is_valid_placement(current_solution, new_pos):
                    continue

                current_solution[new_pos] = best_match_id
                unplaced_pieces.remove(best_match_id)
                self._add_candidates_to_pool(best_match_id, new_pos, candidate_pool, unplaced_pieces, current_solution)
                placed_flag = True
                break # We placed one, loop back to refresh logic if needed
            
            if placed_flag: continue

            # --- PHASE 2: Forced Placement (Stuck?) ---
            # If heap is empty or all heap candidates violated boundaries/collisions
            
            open_slots = self.get_open_boundary_slots(current_solution)
            if not open_slots:
                print("Error: No open slots valid within grid constraints. Puzzle might be unsolvable with this seed.")
                break 

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
                print("Error: Could not find any match to force placement.")
                break

        return [current_solution]

    def _add_candidates_to_pool(self, placed_id, placed_pos, pool, unplaced, current_sol):
        for r in self.relations:
            dx, dy = {0: (1,0), 1: (0,1), 2: (-1,0), 3: (0,-1)}[r]
            target_pos = (placed_pos[0] + dx, placed_pos[1] + dy)
            
            if target_pos in current_sol: continue
            
            # Optimization: Don't add to heap if it already violates dimensions
            # (Note: Checking here saves heap operations, checking in solve() ensures correctness)
            if not self._is_valid_placement(current_sol, target_pos):
                continue

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

    def reconstruct_image(self, solution_dict):
        if not solution_dict: return None
        coords = list(solution_dict.keys())
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        
        min_x, min_y = min(xs), min(ys)
        
        # Since we enforced constraints, output is exactly the grid size
        full_h = self.grid_h * self.piece_size
        full_w = self.grid_w * self.piece_size
        
        canvas = np.zeros((full_h, full_w, 3), dtype=np.uint8)
        
        for pos, pid in solution_dict.items():
            px = (pos[0] - min_x) * self.piece_size
            py = (pos[1] - min_y) * self.piece_size
            
            if px < full_w and py < full_h:
                canvas[py:py+self.piece_size, px:px+self.piece_size] = self.pieces[pid].image
            
        return canvas

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: File not found at {IMAGE_PATH}")
        return

    # Load
    img = cv2.imread(IMAGE_PATH)
    h, w, _ = img.shape
    
    # Calculate grid dimensions based on piece size
    grid_w = w // PIECE_SIZE
    grid_h = h // PIECE_SIZE
    
    new_h = grid_h * PIECE_SIZE
    new_w = grid_w * PIECE_SIZE
    img = img[:new_h, :new_w]
    
    print(f"Processing Image: {new_w}x{new_h}")
    print(f"Grid Dimensions: {grid_w} columns x {grid_h} rows")

    pieces_img = []
    for y in range(0, new_h, PIECE_SIZE):
        for x in range(0, new_w, PIECE_SIZE):
            pieces_img.append(img[y:y+PIECE_SIZE, x:x+PIECE_SIZE])

    np.random.seed(42) 
    np.random.shuffle(pieces_img)

    # Pass constraints to Solver
    solver = PaikinTalSolver(piece_size=PIECE_SIZE, grid_width=grid_w, grid_height=grid_h)
    for p in pieces_img:
        solver.add_piece(p)

    solutions = solver.solve()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if solutions:
        result = solver.reconstruct_image(solutions[0])
        out_path = f"{OUTPUT_DIR}/solution_constrained.jpg"
        cv2.imwrite(out_path, result)
        print(f"Saved constrained solution to: {out_path} ({result.shape[1]}x{result.shape[0]})")

if __name__ == "__main__":
    main()