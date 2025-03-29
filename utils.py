import numpy as np
import io
import warnings
import os
from PIL import Image
import tempfile

def load_image(file_path):
    """
    Attempt to load an image through multiple methods, returning a PIL Image object.

    Args:
        file_path: Path to the image file

    Returns:
        PIL.Image object or None if the image couldn't be loaded
    """
    
    # Specialized loaders
    
    def load_svg(file_path):
        """Convert SVG to PIL Image"""
        try:
            from cairosvg import svg2png
            import tempfile
            
            # Create a temporary PNG file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_filename = tmp.name
            
            # Convert SVG to PNG
            svg2png(url=file_path, write_to=tmp_filename)
            
            # Load the PNG with PIL
            img = Image.open(tmp_filename)
            img.load()
            
            # Clean up
            os.unlink(tmp_filename)
            
            return img
        except Exception as e:
            return None
    
    def load_wmf(file_path):
        """Convert WMF/EMF to PIL Image"""
        try:
            import pymagewell
            
            # Convert to PNG via pymagewell
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_filename = tmp.name
            
            pymagewell.convert(file_path, tmp_filename)
            
            # Load with PIL
            img = Image.open(tmp_filename)
            img.load()
            
            # Clean up
            os.unlink(tmp_filename)
            
            return img
        except Exception as e:
            return None
    
    def load_psd(file_path):
        """Load PSD using psd-tools"""
        try:
            from psd_tools import PSDImage
            
            psd = PSDImage.open(file_path)
            img = psd.composite()  # Compose all layers
            
            return img
        except Exception as e:
            return None
    
    def load_xcf(file_path):
        """Load GIMP XCF files"""
        try:
            import gimpformats.gimpXcfDocument
            
            xcf = gimpformats.gimpXcfDocument.GimpXcfDocument(file_path)
            # Convert the composite image to a PIL Image
            arr = xcf.getCompositeImage()
            img = Image.fromarray(arr)
            
            return img
        except Exception as e:
            return None
    
    def load_heic(file_path):
        """Load HEIC/HEIF images"""
        try:
            import pyheif
            
            # Read HEIC file
            heif_file = pyheif.read(file_path)
            
            # Convert to PIL Image
            img = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            
            return img
        except Exception as e:
            return None
    
    def load_ai_eps(file_path):
        """Load AI/EPS files using Ghostscript"""
        try:
            import ghostscript
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_filename = tmp.name
            
            # Use Ghostscript to convert to PNG
            args = [
                "gs", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m",
                f"-sOutputFile={tmp_filename}", "-r300", file_path
            ]
            ghostscript.Ghostscript(*args)
            
            # Load with PIL
            img = Image.open(tmp_filename)
            img.load()
            
            # Clean up
            os.unlink(tmp_filename)
            
            return img
        except Exception as e:
            return None
    
    def load_exr_hdr(file_path):
        """Load OpenEXR or HDR files"""
        try:
            import OpenEXR
            import Imath
            
            exr_file = OpenEXR.InputFile(file_path)
            dw = exr_file.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            
            # Read the three color channels as 32-bit floats
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            (R, G, B) = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32) for c in 'RGB']
            
            # Reshape and create RGB array
            rgb = np.zeros((size[1], size[0], 3), dtype=np.float32)
            rgb[:, :, 0] = R.reshape(size[1], size[0])
            rgb[:, :, 1] = G.reshape(size[1], size[0])
            rgb[:, :, 2] = B.reshape(size[1], size[0])
            
            # Tone map the HDR image (simple exposure control)
            rgb = np.clip(rgb * 0.25, 0, 1)
            
            # Convert to 8-bit
            rgb_8bit = (rgb * 255).astype(np.uint8)
            img = Image.fromarray(rgb_8bit)
            
            return img
        except Exception as e:
            return None
    
    def load_dds(file_path):
        """Load DirectDraw Surface files"""
        try:
            import wand.image
            
            with wand.image.Image(filename=file_path) as wand_img:
                # Convert to PNG in memory
                wand_img.format = 'png'
                png_blob = wand_img.make_blob()
            
            # Load the PNG blob with PIL
            img = Image.open(io.BytesIO(png_blob))
            
            return img
        except Exception as e:
            return None
    
    def load_tga(file_path):
        """Load TGA files explicitly"""
        try:
            from PIL import TgaImagePlugin  # Explicit import
            
            img = Image.open(file_path)
            img.load()
            
            return img
        except Exception as e:
            return None
    
    def load_raw(file_path):
        """Load RAW camera files"""
        try:
            import rawpy
            
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess()
            
            # Convert numpy array to PIL Image
            img = Image.fromarray(rgb)
            
            return img
        except Exception as e:
            return None
    
    def load_jxl(file_path):
        """Load JPEG XL files"""
        try:
            import jxlpy
            
            # Read the JPEG XL file
            data = jxlpy.JXLDecompressor().decode_file(file_path)
            
            # Convert to PIL Image
            img = Image.fromarray(data)
            
            return img
        except Exception as e:
            return None
    
    def load_video_first_frame(file_path):
        """Extract first frame from video file"""
        try:
            import cv2
            
            # Open the video file
            cap = cv2.VideoCapture(file_path)
            
            # Read the first frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise Exception("Could not read frame")
            
            # Convert from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            img = Image.fromarray(rgb_frame)
            
            return img
        except Exception as e:
            return None
    
    if not os.path.exists(file_path):
        return None
    
    # Get file extension (lowercase)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower().lstrip('.')
    
    def reencode_as_png(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)  # Reset buffer position
        return Image.open(buffer)
    
    # Step 1: Try direct PIL loading (covers most common formats)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress PIL warnings
            img = Image.open(file_path)
            img.load()  # This will verify the image can actually be read
            img = reencode_as_png(img)
            return img
    except Exception as e:
        pass
    
    # Step 2: Try specialized libraries based on file extension
    if ext in ['svg']:
        return load_svg(file_path)
    elif ext in ['wmf', 'emf']:
        return load_wmf(file_path)
    elif ext in ['psd']:
        return load_psd(file_path)
    elif ext in ['xcf']:
        return load_xcf(file_path)
    elif ext in ['heic', 'heif']:
        return load_heic(file_path)
    elif ext in ['ai', 'eps']:
        return load_ai_eps(file_path)
    elif ext in ['exr', 'hdr']:
        return load_exr_hdr(file_path)
    elif ext in ['dds']:
        return load_dds(file_path)
    elif ext in ['tga']:
        return load_tga(file_path)
    elif ext in ['cr2', 'cr3', 'nef', 'arw', 'raw', 'orf', 'rw2', 'dng', 'x3f']:
        return load_raw(file_path)
    elif ext in ['jxl', 'jpxl']:
        return load_jxl(file_path)
    elif ext in ['mp4', 'gif']:
        return load_video_first_frame(file_path)  # For video, we'll extract the first frame
    
    # If we get here, we couldn't load the image
    return None


image_formats = (
    'png',  # Common raster format, lossless compression
    'jpg',  # Common raster format, lossy compression
    'jpeg',  # Same as JPG, just a different extension
    'webp',  # Modern format with better compression than JPG/PNG
    'bmp',  # Uncompressed bitmap format
    'tif',  # High-quality raster format, often used in photography
    'tiff',  # Same as TIF, different extension
    'gif',  # Supports animation, limited to 256 colors
    'mp4',  # Video format, but sometimes used for image sequences
    'ico',  # Windows icon format
    'heic',  # High Efficiency Image Format, used by Apple devices
    'heif',  # Same as HEIC, just a different extension
    'jp2',  # JPEG 2000, better quality than JPEG but less common
    'j2k',  # JPEG 2000 codestream format
    'jpf',  # JPEG 2000, another possible extension
    'svg',  # Scalable Vector Graphics, XML-based
    'eps',  # Encapsulated PostScript, used for vector graphics
    
    'cr2',  # Canon RAW image format
    'cr3',  # Newer Canon RAW format
    'nef',  # Nikon RAW format
    'arw',  # Sony RAW format
    'orf',  # Olympus RAW format
    'rw2',  # Panasonic RAW format
    'dng',  # Adobe Digital Negative, standardized RAW format
    'dds',  # DirectDraw Surface, used in games and textures
    'psd',  # Adobe Photoshop format, supports layers
    'xcf',  # GIMP image format, supports layers
    'tga',  # Targa format, used in older games and design
    'pbm',  # Portable Bitmap, black-and-white images
    'pgm',  # Portable Graymap, grayscale images
    'ppm',  # Portable Pixmap, color images
    'exr',  # OpenEXR, high dynamic range format
    'ai',  # Adobe Illustrator vector format
    'wmf',  # Windows Metafile, vector graphics format
    'emf',  # Enhanced Metafile, improved version of WMF
    'cgm',  # Computer Graphics Metafile, vector graphics format
    'apng',  # Animated PNG, supports full alpha channel
    'avif',  # AV1 Image Format, highly efficient modern format
    'icns',  # macOS icon format
    'cur',  # Windows cursor file format (similar to ICO)
    'mng',  # Multiple-image Network Graphics, like GIF but more advanced
    'flif',  # Free Lossless Image Format, not widely supported
    
    'qoi',  # Quite OK Image Format, simple and lossless
    'jpxl',  # JPEG XR, improved compression over JPEG
    'hif',  # High Efficiency Image File Format (variant of HEIF)
    'jxl',  # JPEG XL, modern replacement for JPEG
    'rla',  # Wavefront RLA, used in 3D rendering
    'rpf',  # Rich Pixel Format, similar to RLA
    'iff',  # Interchange File Format, used on Amiga systems
    'sgi',  # Silicon Graphics Image, used in SGI workstations
    'vda',  # Targa variant
    'icb',  # Targa variant
    'vst',  # Targa variant
    'pix',  # Alias PIX image format
    'blp',  # Blizzard image format (used in games like World of Warcraft)
    'vtf',  # Valve Texture Format, used in Source engine games
    'gbr',  # GIMP brush file (sometimes contains image data)
    'xpm',  # X PixMap, used in X11 environments
    'xwd',  # X Window Dump, used in X11 environments
    'otb',  # Over The Air Bitmap, used in mobile devices
    'btf',  # Binary Texture Format, used in some game engines
    
    'ras',  # Sun Raster Image, used in Unix systems
    'cin',  # Cineon, used in digital film production
    'dpx',  # Digital Picture Exchange, professional film format
    'hdr',  # Radiance HDR, used for high dynamic range images
    'pcx',  # Paintbrush Bitmap, used in older Windows programs
    'pict',  # Macintosh PICT format (legacy)
    'sk1',  # sK1 vector graphics format
    'wbmp',  # Wireless Bitmap, used in mobile devices
    'webp2',  # Experimental WebP 2 format
    'x3f',  # Sigma RAW image format
    'mef',  # Mamiya RAW image format
    'mos',  # Leaf RAW image format
    'pef',  # Pentax RAW image format
    'srw',  # Samsung RAW image format
    'bay',  # Casio RAW image format
    'r3d',  # RED Digital Camera RAW format
    
    'fpx',  # Flashpix, used for photographic images
    'qif',  # Quicktime Image Format, used for video stills
    'gifv',  # GIF Video, sometimes used for animated GIFs with audio
    'raw',  # General term for raw image formats
    'lbm',  # IFF ILBM, used in older Amiga systems
    'pvr',  # PowerVR Texture File, used in gaming
    'tim',  # TIM (used in PlayStation graphics)
    'xif',  # Xerox Imaging Format
    'liff',  # Lossless Image File Format, used by some scanners
    'raw16',  # 16-bit raw image format, sometimes used for high-dynamic-range images
    'tiff_lzw',  # TIFF with LZW compression (common variant)
)


from PyQt5.QtWidgets import QApplication, QSplashScreen, QProgressBar, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QFont


class SplashScreen(QSplashScreen):
    """Custom splash screen with manually updated progress bar and status text"""
    
    def __init__(self, app_name="Application", logo_path=None, width=400, height=320):
        super().__init__()
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.FramelessWindowHint)
        
        # Create the content widget
        self.content = QWidget()
        layout = QVBoxLayout(self.content)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title label
        self.title_label = QLabel(app_name)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.title_label.setStyleSheet("color: #333333;")
        
        # Loading label
        self.loading_label = QLabel("Loading...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setFont(QFont("Arial", 10))
        self.loading_label.setStyleSheet("color: #666666;")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(15)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                background-color: #F5F5F5;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4A86E8;
                border-radius: 5px;
            }
        """)
        
        # Status label (for showing current asset being loaded)
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 9))
        self.status_label.setStyleSheet("color: #666666;")
        self.status_label.setWordWrap(True)
        
        # Add logo if provided
        if logo_path:
            logo = QLabel()
            pixmap = QPixmap(logo_path)
            logo.setPixmap(pixmap.scaled(QSize(80, 80), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            logo.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo)
            layout.addSpacing(10)
        
        # Add widgets to layout
        layout.addWidget(self.title_label)
        layout.addSpacing(10)
        layout.addWidget(self.loading_label)
        layout.addSpacing(20)
        layout.addWidget(self.progress_bar)
        layout.addSpacing(5)
        layout.addWidget(self.status_label)
        
        # Create a blank pixmap with the desired size
        self.pixmap = QPixmap(width, height)
        self.pixmap.fill(Qt.white)
        self.setPixmap(self.pixmap)
        self.setFixedSize(width, height)
        
        # Center in screen
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - width) // 2
        y = (screen_geometry.height() - height) // 2
        self.move(x, y)
    
    def drawContents(self, painter):
        """Override to draw the content widget on the splash screen"""
        try:
            self.content.setGeometry(0, 0, self.width(), self.height())
            self.content.render(painter)
        except:pass
    
    def set_progress(self, current, total=100, status_text=""):
        """
        Update the progress bar value based on current progress and total
        Optionally update the status text to show what's being loaded
        """
        percentage = int((current / total) * 100)
        self.progress_bar.setValue(percentage)
        
        # Update status text if provided
        if status_text:
            self.status_label.setText(status_text)
        
        self.repaint()  # Force immediate update
        QApplication.processEvents()  # Process pending events to keep UI responsive
    
    def set_loading_text(self, text):
        """Update the loading text"""
        self.loading_label.setText(text)
        self.repaint()
        QApplication.processEvents()


def truncate_path(path, max_length):
    if len(path) <= max_length:
        return path
    return "..." + path[-(max_length - 3):]