import sys
import numpy as np
import open3d as o3d
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

class PointCloudApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Cloud Viewer")
        self.setGeometry(300, 300, 400, 200)

        self.label = QLabel("Click the button below to open a .npy point cloud file", self)
        self.label.setAlignment(Qt.AlignCenter)

        self.button = QPushButton("Open .npy File")
        self.button.clicked.connect(self.open_file_dialog)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open .npy File", "", "NumPy files (*.npy)")
        if file_path:
            self.load_and_show_point_cloud(file_path)

    def load_and_show_point_cloud(self, file_path):
        try:
            points = np.load(file_path)
            if points.ndim != 2 or points.shape[1] != 3:
                self.label.setText("Invalid point cloud format! Expected shape (N, 3)")
                return

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.visualization.draw_geometries([pcd])
            self.label.setText("Loaded and displayed successfully. You can load another file.")
        except Exception as e:
            self.label.setText(f"Error: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PointCloudApp()
    window.show()
    sys.exit(app.exec_())