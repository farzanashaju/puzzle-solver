import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
import time
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern

@dataclass
# represents a single puzzle piece
class PuzzlePiece:
    id: int
    # rgb image patch
    image: np.ndarray
    # ground truth location
    # (row, col)
    true_position: Tuple[int, int]
    # features filled after feature extraction
    features: dict
    # used during assembly
    current_position: Optional[Tuple[int, int]] = None
    # used during assembly
    is_placed: bool = False

# extract features from puzzle pieces
class FeatureExtractor:
    # number of rows/cols used at each piece side to compute boundary features
    def __init__(self, boundary_width=5):
        self.boundary_width = boundary_width
    
    # extract a small strip on one side of width boundary_width
    # compute features for that strip
    def extract_boundary_features(self, image: np.ndarray, side: str) -> np.ndarray:
        h, w = image.shape[:2]
        bw = self.boundary_width
        
        # extract boundary region
        if side == 'top':
            boundary = image[:bw, :]
        elif side == 'bottom':
            boundary = image[-bw:, :]
        elif side == 'left':
            boundary = image[:, :bw]
        elif side == 'right':
            boundary = image[:, -bw:]
        else:
            raise ValueError(f"Invalid side: {side}")
        
        features = []
        
        # rgb statistics
        # mean, std, median per channel
        for channel in range(3):
            channel_data = boundary[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data)
            ])
        
        # convert the boundary to grayscale and compute sobel gradients
        gray_boundary = cv2.cvtColor(boundary, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray_boundary, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_boundary, cv2.CV_64F, 0, 1, ksize=3)
        
        features.extend([
            np.mean(np.abs(grad_x)),
            np.std(grad_x),
            np.mean(np.abs(grad_y)),
            np.std(grad_y)
        ])
        
        # local binary pattern histogram
        # captures texture information
        lbp = local_binary_pattern(gray_boundary, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.flatten(), bins=10, range=(0, 10))
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)
        features.extend(lbp_hist)
        
        return np.array(features)
    
    # extract global patch features
    def extract_patch_features(self, image: np.ndarray) -> np.ndarray:
        features = []
        
        # colour statistics
        # mean and std for each channel
        for channel in range(3):
            channel_data = image[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data)
            ])
        
        # dominant colors using k-means
        pixels = image.reshape(-1, 3).astype(np.float32)
        # cluster into 3 groups
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
        kmeans.fit(pixels)
        # cluster centers = dominant color vectors
        dominant_colors = kmeans.cluster_centers_
        
        features.extend(dominant_colors.flatten())
        
        # texture features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # std deviation of grayscale image
        features.append(np.std(gray))
        
        return np.array(features)
    
    # extract all features for a puzzle piece
    def extract_all_features(self, piece: PuzzlePiece) -> dict:
        image = piece.image
        
        features = {
            'top': self.extract_boundary_features(image, 'top'),
            'bottom': self.extract_boundary_features(image, 'bottom'),
            'left': self.extract_boundary_features(image, 'left'),
            'right': self.extract_boundary_features(image, 'right'),
            'patch': self.extract_patch_features(image),
            # for training only
            'position': np.array(piece.true_position)
        }
        
        return features

# produce a compatibility score for a candidate adjacency
class KNNCompatibilityModel:
    
    def __init__(self, n_neighbors=7, metric='euclidean', use_pca=True, pca_components=20):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.use_pca = use_pca
        self.pca_components = pca_components
        
        # separate models for horizontal vs vertical adjacencies
        self.models = {
            # left-right
            'horizontal': None,
            # top-bottom
            'vertical': None
        }
        
        self.pca_models = {
            'horizontal': None,
            'vertical': None
        }
        
        self.training_data = {
            'horizontal': {'X': [], 'y': []},
            'vertical': {'X': [], 'y': []}
        }
    
    # create feature vector for a pair of pieces
    def create_pair_features(self, feat1: dict, feat2: dict, direction: str) -> np.ndarray:
        if direction == 'horizontal':
            # piece1 is left, piece2 is right
            boundary1 = feat1['right']
            boundary2 = feat2['left']
        else:
            # piece1 is top, piece2 is bottom
            boundary1 = feat1['bottom']
            boundary2 = feat2['top']
        
        # concatenate boundary features and patch features
        pair_features = np.concatenate([
            boundary1,
            boundary2,
            feat1['patch'],
            feat2['patch'],
            # difference features
            np.abs(boundary1 - boundary2)
        ])
        
        return pair_features
    
    # train knn models on both positive (adjacent) and negative (non-adjacent) pairs
    def train(self, pieces: List[PuzzlePiece], negative_ratio=2.0):
        
        # extract features for all pieces
        extractor = FeatureExtractor()
        for piece in pieces:
            piece.features = extractor.extract_all_features(piece)
        
        # number of rows and columns
        n_rows = max(p.true_position[0] for p in pieces) + 1
        n_cols = max(p.true_position[1] for p in pieces) + 1
        
        # create position lookup
        pos_to_piece = {p.true_position: p for p in pieces}
        
        # positive horizontal pairs (left-right)
        for r in range(n_rows):
            for c in range(n_cols - 1):
                if (r, c) in pos_to_piece and (r, c + 1) in pos_to_piece:
                    p1 = pos_to_piece[(r, c)]
                    p2 = pos_to_piece[(r, c + 1)]
                    features = self.create_pair_features(p1.features, p2.features, 'horizontal')
                    self.training_data['horizontal']['X'].append(features)
                    self.training_data['horizontal']['y'].append(1) # compatible
        
        # positive vertical pairs (top-bottom)
        for r in range(n_rows - 1):
            for c in range(n_cols):
                if (r, c) in pos_to_piece and (r + 1, c) in pos_to_piece:
                    p1 = pos_to_piece[(r, c)]
                    p2 = pos_to_piece[(r + 1, c)]
                    features = self.create_pair_features(p1.features, p2.features, 'vertical')
                    self.training_data['vertical']['X'].append(features)
                    self.training_data['vertical']['y'].append(1) # compatible
        
        # sample non-adjacent pairs for negative examples
        num_pos_horizontal = len(self.training_data['horizontal']['y'])
        num_pos_vertical = len(self.training_data['vertical']['y'])
        num_neg_horizontal = int(num_pos_horizontal * negative_ratio)
        num_neg_vertical = int(num_pos_vertical * negative_ratio)
        
        # negative horizontal pairs
        neg_count = 0
        while neg_count < num_neg_horizontal:
            p1, p2 = np.random.choice(pieces, 2, replace=False)
            r1, c1 = p1.true_position
            r2, c2 = p2.true_position
            if not (r1 == r2 and abs(c1 - c2) == 1):
                features = self.create_pair_features(p1.features, p2.features, 'horizontal')
                self.training_data['horizontal']['X'].append(features)
                self.training_data['horizontal']['y'].append(0) # not compatible
                neg_count += 1
        
        # negative vertical pairs
        neg_count = 0
        while neg_count < num_neg_vertical:
            p1, p2 = np.random.choice(pieces, 2, replace=False)
            r1, c1 = p1.true_position
            r2, c2 = p2.true_position
            if not (c1 == c2 and abs(r1 - r2) == 1):
                features = self.create_pair_features(p1.features, p2.features, 'vertical')
                self.training_data['vertical']['X'].append(features)
                self.training_data['vertical']['y'].append(0) # not compatible
                neg_count += 1

        # train models
        for direction in ['horizontal', 'vertical']:
            X = np.array(self.training_data[direction]['X'])
            y = np.array(self.training_data[direction]['y'])
            
            if self.use_pca and X.shape[0] > self.pca_components:
                pca = PCA(n_components=min(self.pca_components, X.shape[0], X.shape[1]))
                X = pca.fit_transform(X)
                self.pca_models[direction] = pca
            
            model = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
            model.fit(X)
            self.models[direction] = model
            
            # store labels
            self.training_data[direction]['y'] = y
            self.training_data[direction]['X_transformed'] = X
    
    # get compatibility score between two pieces
    def get_compatibility(self, piece1: PuzzlePiece, piece2: PuzzlePiece, direction: str) -> float:
        pair_features = self.create_pair_features(piece1.features, piece2.features, direction)
        pair_features = pair_features.reshape(1, -1)
        
        if self.pca_models[direction] is not None:
            pair_features = self.pca_models[direction].transform(pair_features)
        
        distances, indices = self.models[direction].kneighbors(pair_features)
        
        # use labels of nearest neighbors to compute compatibility
        # score based on proportion of positive neighbors and their distances
        neighbor_labels = self.training_data[direction]['y'][indices[0]]
        positive_ratio = np.sum(neighbor_labels) / len(neighbor_labels)
        
        # weight by inverse distance
        # closer positive examples should count more
        weights = 1.0 / (distances[0] + 1e-6)
        # normalize
        weights = weights / np.sum(weights)
        weighted_positive_ratio = np.sum(weights * neighbor_labels)
        
        # combine both metrics
        compatibility = 0.6 * weighted_positive_ratio + 0.4 * positive_ratio
        
        return compatibility

# assemble puzzle pieces using a greedy algorithm with backtracking
class PuzzleAssembler:    
    def __init__(self, knn_model: KNNCompatibilityModel, n_rows: int, n_cols: int):
        self.knn_model = knn_model
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.grid = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    
    # reset the grid and all piece placements
    def reset_grid(self):
        self.grid = [[None for _ in range(self.n_cols)] for _ in range(self.n_rows)]
    
    # check if placing a piece at a position is valid
    # return compatibilty score
    def is_valid_placement(self, piece: PuzzlePiece, row: int, col: int, 
                          placed_pieces: List[PuzzlePiece]) -> Tuple[bool, float]:
        if self.grid[row][col] is not None:
            return False, 0.0
        
        total_compatibility = 0.0
        num_neighbors = 0
        
        # check left neighbor
        if col > 0 and self.grid[row][col - 1] is not None:
            left_piece = self.grid[row][col - 1]
            comp = self.knn_model.get_compatibility(left_piece, piece, 'horizontal')
            total_compatibility += comp
            num_neighbors += 1
        
        # check right neighbor
        if col < self.n_cols - 1 and self.grid[row][col + 1] is not None:
            right_piece = self.grid[row][col + 1]
            comp = self.knn_model.get_compatibility(piece, right_piece, 'horizontal')
            total_compatibility += comp
            num_neighbors += 1
        
        # check top neighbor
        if row > 0 and self.grid[row - 1][col] is not None:
            top_piece = self.grid[row - 1][col]
            comp = self.knn_model.get_compatibility(top_piece, piece, 'vertical')
            total_compatibility += comp
            num_neighbors += 1
        
        # check bottom neighbor
        if row < self.n_rows - 1 and self.grid[row + 1][col] is not None:
            bottom_piece = self.grid[row + 1][col]
            comp = self.knn_model.get_compatibility(piece, bottom_piece, 'vertical')
            total_compatibility += comp
            num_neighbors += 1
        
        avg_compatibility = total_compatibility / num_neighbors if num_neighbors > 0 else 0.5
        
        return True, avg_compatibility
    
    # place a piece on the grid
    def place_piece(self, piece: PuzzlePiece, row: int, col: int):
        self.grid[row][col] = piece
        piece.current_position = (row, col)
        piece.is_placed = True
    
    # remove a piece from the grid
    def remove_piece(self, row: int, col: int):
        if self.grid[row][col] is not None:
            self.grid[row][col].is_placed = False
            self.grid[row][col].current_position = None
            self.grid[row][col] = None
    
    # starting strategies: random, center fixed, boundary fixed, random_fixed
    # pick the next best local move each time
    def greedy_assemble(self, pieces: List[PuzzlePiece], strategy='random', num_random_fixed=5) -> List[List[PuzzlePiece]]:
        
        # reset all pieces
        for piece in pieces:
            piece.is_placed = False
            piece.current_position = None
        
        unplaced = list(pieces)
        
        if strategy == 'center_fixed':
            # find and place the true center piece
            center_row, center_col = self.n_rows // 2, self.n_cols // 2
            center_piece = next(p for p in pieces if p.true_position == (center_row, center_col))
            self.place_piece(center_piece, center_row, center_col)
            unplaced.remove(center_piece)
        
        elif strategy == 'boundary_fixed':
            # place all boundary pieces at their correct positions
            boundary_count = 0
            for piece in pieces:
                r, c = piece.true_position
                # check if boundary piece
                if r == 0 or r == self.n_rows - 1 or c == 0 or c == self.n_cols - 1:
                    self.place_piece(piece, r, c)
                    unplaced.remove(piece)
                    boundary_count += 1
        
        elif strategy == 'random_fixed':
            # place a random selection of pieces at their correct positions
            num_to_fix = min(num_random_fixed, len(pieces))
            fixed_pieces = np.random.choice(pieces, num_to_fix, replace=False)
            
            for piece in fixed_pieces:
                r, c = piece.true_position
                self.place_piece(piece, r, c)
                unplaced.remove(piece)
                    
        else:
            # start with a random piece in the center
            start_piece = unplaced[0]
            start_row, start_col = self.n_rows // 2, self.n_cols // 2
            self.place_piece(start_piece, start_row, start_col)
            unplaced.remove(start_piece)
        
        # continue all are placed
        iteration = 0
        max_iterations = len(pieces) * 10
        
        while unplaced and iteration < max_iterations:
            iteration += 1
            best_score = -1
            best_piece = None
            best_position = None
            
            # try each unplaced piece at each empty position
            for piece in unplaced:
                for r in range(self.n_rows):
                    for c in range(self.n_cols):
                        valid, score = self.is_valid_placement(piece, r, c, unplaced)
                        if valid and score > best_score:
                            best_score = score
                            best_piece = piece
                            best_position = (r, c)
            
            if best_piece is not None:
                self.place_piece(best_piece, best_position[0], best_position[1])
                unplaced.remove(best_piece)
                
            else:
                # no valid placement found, place randomly
                if unplaced:
                    for r in range(self.n_rows):
                        for c in range(self.n_cols):
                            if self.grid[r][c] is None:
                                self.place_piece(unplaced[0], r, c)
                                unplaced.pop(0)
                                break
                        if not unplaced or self.grid[r][c] is not None:
                            break
        
        return self.grid


class JigsawPuzzleSolver:
    def __init__(self, n_rows=8, n_cols=8, n_neighbors=7):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.pieces = []
        self.knn_model = KNNCompatibilityModel(n_neighbors=n_neighbors)
        self.assembler = None
    
    # create puxxle pieces from an image
    def create_puzzle(self, image_path: str, shuffle=True) -> List[PuzzlePiece]:
        # Load image
        image = cv2.imread(str(image_path))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # calculate piece dimensions
        piece_h = h // self.n_rows
        piece_w = w // self.n_cols
        
        # resize image to exact multiple
        new_h = piece_h * self.n_rows
        new_w = piece_w * self.n_cols
        image = cv2.resize(image, (new_w, new_h))
        
        # create pieces
        pieces = []
        piece_id = 0
        
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                y1, y2 = r * piece_h, (r + 1) * piece_h
                x1, x2 = c * piece_w, (c + 1) * piece_w
                piece_img = image[y1:y2, x1:x2].copy()
                
                piece = PuzzlePiece(
                    id=piece_id,
                    image=piece_img,
                    true_position=(r, c),
                    features={}
                )
                pieces.append(piece)
                piece_id += 1
        
        if shuffle:
            np.random.shuffle(pieces)
        
        self.pieces = pieces
        self.original_image = image
        
        return pieces
    
    # solve the puzzle using multiple strategies and return all solutions
    def solve(self, random_fixed_counts=[5, 10, 15]) -> dict:
        start_time = time.time()
        
        # train knn model
        self.knn_model.train(self.pieces)
        
        # try different starting strategies
        strategies = ['random', 'center_fixed', 'boundary_fixed']
        
        # add random_fixed strategies with different counts
        for count in random_fixed_counts:
            strategies.append(f'random_fixed_{count}')
        
        solutions = {}
        
        self.assembler = PuzzleAssembler(self.knn_model, self.n_rows, self.n_cols)
        
        for strategy_name in strategies:
            # reset grid for new strategy
            self.assembler.reset_grid()
            
            # parse strategy name and parameters
            if strategy_name.startswith('random_fixed_'):
                num_fixed = int(strategy_name.split('_')[-1])
                solution = self.assembler.greedy_assemble(self.pieces, strategy='random_fixed', num_random_fixed=num_fixed)
            else:
                solution = self.assembler.greedy_assemble(self.pieces, strategy=strategy_name)
            
            solutions[strategy_name] = solution
        
        elapsed = time.time() - start_time
        
        return solutions
    
    # reconstrut image from solution grid
    def reconstruct_image(self, solution: List[List[PuzzlePiece]]) -> np.ndarray:
        if not solution or not solution[0]:
            return None
        
        piece_h, piece_w = solution[0][0].image.shape[:2]
        result = np.zeros((self.n_rows * piece_h, self.n_cols * piece_w, 3), dtype=np.uint8)
        
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if solution[r][c] is not None:
                    y1, y2 = r * piece_h, (r + 1) * piece_h
                    x1, x2 = c * piece_w, (c + 1) * piece_w
                    result[y1:y2, x1:x2] = solution[r][c].image
        
        return result
    
    # evaluation solution
    def evaluate(self, solution: List[List[PuzzlePiece]]) -> dict:
        metrics = {
            'direct_accuracy': 0.0,
            'neighbor_accuracy': 0.0,
            'ssim': 0.0
        }
        
        total_pieces = self.n_rows * self.n_cols
        correct_positions = 0
        total_neighbors = 0
        correct_neighbors = 0
        
        # direct accuracy
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if solution[r][c] is not None:
                    if solution[r][c].true_position == (r, c):
                        correct_positions += 1
        
        metrics['direct_accuracy'] = correct_positions / total_pieces
        
        # neighbor accuracy
        # check only right and bottom to avoid double counting
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if solution[r][c] is not None:
                    true_r, true_c = solution[r][c].true_position
                    
                    # check right neighbor
                    if c < self.n_cols - 1 and solution[r][c + 1] is not None:
                        true_r2, true_c2 = solution[r][c + 1].true_position
                        if true_r == true_r2 and true_c + 1 == true_c2:
                            correct_neighbors += 1
                        total_neighbors += 1
                    
                    # check bottom neighbor
                    if r < self.n_rows - 1 and solution[r + 1][c] is not None:
                        true_r2, true_c2 = solution[r + 1][c].true_position
                        if true_r + 1 == true_r2 and true_c == true_c2:
                            correct_neighbors += 1
                        total_neighbors += 1
        
        if total_neighbors > 0:
            metrics['neighbor_accuracy'] = correct_neighbors / total_neighbors
        
        # ssim (structural similarity index)
        # measures visual similarity between the reconstructed and original image
        reconstructed = self.reconstruct_image(solution)
        if reconstructed is not None:
            gray_orig = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            gray_recon = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2GRAY)
            metrics['ssim'] = ssim(gray_orig, gray_recon, data_range=255)
        
        return metrics
    
    # visualise original and reconstructed images side by side
    def visualize(self, solution: List[List[PuzzlePiece]], save_path: Optional[str] = None):
        reconstructed = self.reconstruct_image(solution)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(self.original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(reconstructed)
        axes[1].set_title('Reconstructed Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close(fig)

def main():
    
    # suppress numpy warnings
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    # find images in dataset directory
    dataset_dir = Path('dataset')
    
    # get all jpg and png images
    image_files = sorted(list(dataset_dir.glob('*.jpg')) + list(dataset_dir.glob('*.png')))
    
    if not image_files:
        print(f"No images found in {dataset_dir}. Please run download_dataset.py first.")
        return
    
    # configure random_fixed strategy counts to test
    random_fixed_counts = [3, 6, 9]  # test with 3, 6, and 9 randomly fixed pieces
    
    # prepare CSV file for metrics (saved inside output dir)
    csv_path = output_dir / 'experiment_metrics.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # write header
    csv_writer.writerow([
        'image_name', 'strategy', 'direct_accuracy', 
        'neighbor_accuracy', 'ssim', 'is_best'
    ])
    
    # process each image
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
        print("-" * 60)
        
        # create and solve puzzle
        solver = JigsawPuzzleSolver(n_rows=6, n_cols=6, n_neighbors=5)
        
        try:
            # create puzzle
            pieces = solver.create_puzzle(str(image_path), shuffle=True)
            
            # solve puzzle with all strategies
            solutions = solver.solve(random_fixed_counts=random_fixed_counts)
            
            # evaluate each strategy
            best_strategy = None
            best_score = -1
            best_solution = None
            all_results = []
            
            for strategy_name, solution in solutions.items():
                print(f"\nEVALUATION FOR '{strategy_name}' STRATEGY")
                metrics = solver.evaluate(solution)
                print(f"Direct Accuracy: {metrics['direct_accuracy']:.2%}")
                print(f"Neighbor Accuracy: {metrics['neighbor_accuracy']:.2%}")
                print(f"SSIM: {metrics['ssim']:.4f}")
                
                # store results
                all_results.append({
                    'strategy': strategy_name,
                    'metrics': metrics,
                    'solution': solution
                })
                
                # track best strategy by neighbor accuracy
                if metrics['neighbor_accuracy'] > best_score:
                    best_score = metrics['neighbor_accuracy']
                    best_strategy = strategy_name
                    best_solution = solution
            
            # write all results to CSV
            for result in all_results:
                is_best = 1 if result['strategy'] == best_strategy else 0
                csv_writer.writerow([
                    image_path.name,
                    result['strategy'],
                    f"{result['metrics']['direct_accuracy']:.4f}",
                    f"{result['metrics']['neighbor_accuracy']:.4f}",
                    f"{result['metrics']['ssim']:.4f}",
                    is_best
                ])
            
            # save only the best strategy visualization
            if best_solution is not None:
                result_path = output_dir / f'best_{image_path.stem}.png'
                solver.visualize(best_solution, save_path=str(result_path))
                print(f"\nBEST STRATEGY: '{best_strategy}' with neighbor accuracy: {best_score:.2%}")
                print(f"Saved visualization: {result_path.name}")
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # close CSV file
    csv_file.close()
    print(f"\n{'='*60}")
    print(f"All metrics saved to: {csv_path}")
    print(f"{'='*60}")
        
if __name__ == "__main__":
    main()