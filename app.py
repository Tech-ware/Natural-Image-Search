import sys
import os
import pickle
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QGridLayout, QScrollArea,
    QMessageBox, QProgressBar, QStatusBar
)
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image

class IndexingThread(QThread):
    progress_updated = pyqtSignal(int)
    indexing_complete = pyqtSignal(torch.Tensor)

    def __init__(self, image_paths, processor, model, device):
        super().__init__()
        self.image_paths = image_paths
        self.processor = processor
        self.model = model
        self.device = device

    def run(self):
        embeddings = []
        total_images = len(self.image_paths)
        for idx, img_path in enumerate(self.image_paths):
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    embedding = self.model.get_image_features(**inputs)
                    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
                embeddings.append(embedding.cpu())
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
            progress = int((idx + 1) / total_images * 100)
            self.progress_updated.emit(progress)
        if embeddings:
            image_embeddings = torch.vstack(embeddings)
            self.indexing_complete.emit(image_embeddings)
        else:
            self.indexing_complete.emit(None)

class ImageSearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Natural Image Search (NIS-v1.0)")
        self.setWindowIcon(QIcon('icon.ico'))
        self.setMinimumSize(800, 600)

        # Initialize attributes
        self.model, self.processor = self.load_model()
        self.image_embeddings = None
        self.image_paths = []
        self.cache_file = ''
        self.image_dir = ''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.init_ui()

    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Scroll area for images
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.grid_layout = QGridLayout()
        scroll_widget.setLayout(self.grid_layout)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        # Search bar layout
        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Describe the image you are looking for")
        self.search_bar.textChanged.connect(self.update_search_results)
        search_layout.addWidget(self.search_bar)

        # Settings button
        settings_button = QPushButton("Select the desired folder")
        settings_button.clicked.connect(self.open_settings)
        search_layout.addWidget(settings_button)
        main_layout.addLayout(search_layout)

        # Status bar for progress
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Apply dark theme
        self.apply_dark_theme()

    def load_model(self):
        try:
            # Load CLIP model and processor
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            return model, processor
        except Exception as e:
            QMessageBox.critical(self, "Model Loading Error", f"Failed to load model: {e}")
            sys.exit(1)

    def open_settings(self):
        # Open directory selection dialog
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dir_path:
            self.image_dir = dir_path
            self.cache_file = os.path.join(self.image_dir, "embeddings_cache.pkl")
            self.load_images()
            if not self.image_paths:
                return
            self.index_images()

    def load_images(self):
        # Load image paths, including subdirectories
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        self.image_paths = []
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(supported_formats):
                    self.image_paths.append(os.path.join(root, file))
        if not self.image_paths:
            QMessageBox.warning(self, "No Images Found", "The selected directory contains no supported image files.")
            return

    def index_images(self):
        # Start indexing in a separate thread
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.image_embeddings = pickle.load(f)
                self.display_images(self.image_paths)
                return
            except Exception as e:
                QMessageBox.warning(self, "Cache Loading Error", f"Failed to load cache: {e}")
                # Continue to re-index images

        self.progress_bar = QProgressBar()
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.progress_bar.setValue(0)

        self.indexing_thread = IndexingThread(
            image_paths=self.image_paths,
            processor=self.processor,
            model=self.model,
            device=self.device
        )
        self.indexing_thread.progress_updated.connect(self.progress_bar.setValue)
        self.indexing_thread.indexing_complete.connect(self.on_indexing_complete)
        self.indexing_thread.start()

    def on_indexing_complete(self, image_embeddings):
        self.progress_bar.setValue(100)
        self.statusBar().removeWidget(self.progress_bar)
        if image_embeddings is None:
            QMessageBox.warning(self, "Indexing Failed", "Failed to index images.")
            return
        self.image_embeddings = image_embeddings
        # Save embeddings to cache
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.image_embeddings, f)
        except Exception as e:
            QMessageBox.warning(self, "Cache Saving Error", f"Failed to save cache: {e}")
        # Initial display of images
        self.display_images(self.image_paths)

    def update_search_results(self, text):
        if not text or self.image_embeddings is None:
            self.display_images(self.image_paths)
            return
        try:
            # Compute text embedding
            inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
            with torch.no_grad():
                text_embedding = self.model.get_text_features(**inputs)
                text_embedding = text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)

            # Compute similarities
            similarities = (self.image_embeddings @ text_embedding.T).squeeze(1)
            sorted_indices = similarities.argsort(descending=True)
            sorted_paths = [self.image_paths[i] for i in sorted_indices]

            # Update displayed images
            self.display_images(sorted_paths)
        except Exception as e:
            QMessageBox.warning(self, "Search Error", f"An error occurred during search: {e}")

    def display_images(self, image_paths):
        # Clear grid layout
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Display images in grid
        cols = 4  # Number of columns in the grid
        for index, img_path in enumerate(image_paths):
            row = index // cols
            col = index % cols
            label = self.create_thumbnail_label(img_path)
            self.grid_layout.addWidget(label, row, col)

    def create_thumbnail_label(self, img_path):
        try:
            pixmap = QPixmap(img_path).scaled(
                200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            pixmap = QPixmap(200, 200)
            pixmap.fill(Qt.GlobalColor.red)
        label = QLabel()
        label.setPixmap(pixmap)
        label.mousePressEvent = lambda event: self.show_full_image(img_path)
        return label

    def show_full_image(self, img_path):
        # Display full-resolution image with close button
        self.full_image_window = QWidget()
        self.full_image_window.setWindowTitle("Image Preview")
        vbox = QVBoxLayout()
        self.full_image_window.setLayout(vbox)
        pixmap = QPixmap(img_path)
        screen_rect = QApplication.primaryScreen().availableGeometry()
        screen_size = screen_rect.size()
        pixmap = pixmap.scaled(
            screen_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(label)

        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.full_image_window.close)
        close_button.setFixedSize(100, 40)
        vbox.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.full_image_window.showFullScreen()

    def apply_dark_theme(self):
        # Apply a dark theme stylesheet
        dark_stylesheet = """
            QWidget {
                background-color: #2b2b2b;
                color: #d3d3d3;
            }
            QLineEdit, QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #555;
                padding: 5px;
                color: #d3d3d3;
            }
            QLabel {
                color: #d3d3d3;
            }
            QProgressBar {
                background-color: #555;
                border: 1px solid #555;
                color: #d3d3d3;
                text-align: center;
            }
            QScrollArea {
                border: none;
            }
        """
        self.setStyleSheet(dark_stylesheet)

def main():
    app = QApplication(sys.argv)
    window = ImageSearchApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
