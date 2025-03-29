from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
import sys

# enable dpi scaling for pyqt5
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

from utils import SplashScreen

app = QApplication(sys.argv)

splash = SplashScreen(app_name="Local OCR Indexer -\n- and Search Engine.\n\nDigesting Fresh Files...")
splash.show()

splash.set_progress(0, 100, 'Initilizing packages...')
print('Initilizing packages')
import os
os.environ['RAY_DEDUP_LOGS'] = '0'
import zhmiscellany
splash.set_progress(20, 100, 'Initilizing packages...')
import time
from collections import defaultdict
import threading
import random
from PIL import Image

import zhmiscellanyocr, humanize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import image_formats, load_image

from flask import Flask, render_template_string
from io import BytesIO
import base64

from rapidfuzz import fuzz

def path_to_text_pipeline(path, index=1, group_index=1):
    def timeout(timeout):
        def _timeout(timeout):
            time.sleep(timeout)
            zhmiscellany.misc.die()
        threading.Thread(target=_timeout, args=(timeout,)).start()
    
    timeout(30)
    
    img = load_image(path)
    if img is None:
        return
    
    text = zhmiscellanyocr.ocr(img, config="--psm 11 --oem 3 -c preserve_interword_spaces=1")
    
    if index or total:
        sys.stdout.write(f'|{sig_string}i{index}|{sig_string}g{group_index}|')
    return (path, text)


#splash.set_progress(0, 100, 'Waiting on ray to init...')

print('Waiting on ray to init')
from zhmiscellany._processing_supportfuncs import _ray_init_thread; _ray_init_thread.join()
print('Creating GFI')

splash.set_progress(40, 100, 'Indexing fresh files...')

files = zhmiscellany.fileio.list_files_recursive_cache_optimised_multiprocessed('C:\\', show_timings=True)

splash.set_progress(60, 100, 'Filtering files...')

print('Filtering files')
dot_image_formats = tuple(['.' + format for format in image_formats])

format_paths = defaultdict(list)

for file in files:
    if file.lower().endswith(dot_image_formats):
        ext = file.split('.')[-1].lower()
        
        format_paths[ext].append(file)


db_folder = 'GFI_image_text'
file_name = 'chunk_file'

zhmiscellany.fileio.create_folder(db_folder)
os.chdir(db_folder)

splash.set_progress(70, 100, 'Reading pickle cluster...')

# read existing data
print('Reading data')
files_text = {}
for data_file in zhmiscellany.fileio.abs_listdir('.'):
    if file_name in data_file:
        files_text.update(zhmiscellany.fileio.load_object_from_file(data_file))


# final gathering of files before assigning tasks
total = 0
task_files = []
for key, value in list(format_paths.items()):
    for i, file in enumerate(value):
        if file not in files_text:
            if os.path.exists(file):
                if os.path.getsize(file) > 700:  # can filter out alot of files by assuming a file smaller than 700 bytes can't contain any OCRable text
                    if '\\temp\\' not in file.lower():
                        task_files.append(file)
                        total += 1


if task_files:
    
    splash.set_progress(90, 100, 'Creating tasks...')
    
    print('Creating tasks')
    
    from utils import truncate_path
    
    class Capture_Console:
        def __init__(self):
            self.sig_string = zhmiscellany.string.get_universally_unique_string()
            self.peak_index = 0
        
        def write(self, data):
            if self.sig_string in data:
                strings = data.split('|')
                group_index = 1
                index = 1
                for st in strings:
                    if self.sig_string in st:
                        st = st.replace(self.sig_string, '')
                        match st[0]:
                            case 'i':
                                index = int(st.split('i').pop())
                            case 'g':
                                group_index = int(st.split('g').pop())
                index = max(1, index, self.peak_index)
                self.peak_index = max(index, self.peak_index)
                eta = humanize.precisedelta((((time.time()-start_time)/index)*total)-(time.time()-start_time))
                bsn='\n';print(f'{bsn*10}Completed:{bsn}{zhmiscellany.math.smart_percentage(index, total)}%{bsn}{index}/{total}{bsn}{group_index}/{total_groups}{bsn*2}Files\s:{bsn}{round(index/(time.time()-start_time), 2)}{bsn*2}ETA:{bsn}{eta}')
                index = min(index, total-1)
                splash.set_progress(self.peak_index, total, f'{truncate_path(task_files[index], 60)}\nETA: {eta}.')
            else:
                sys.__stdout__.write(data)  # Print normally
        
        def flush(self):
            sys.__stdout__.flush()
    
    cc = Capture_Console()
    sys.stdout = cc
    sig_string = cc.sig_string
    
    random.shuffle(task_files)
    # creating tasks
    tasks = []
    for i, file in enumerate(task_files, start=1):
        tasks.append((path_to_text_pipeline, [file, i, 0]))
    
    start_time = time.time()
    
    # grouping tasks for better fault tolerance
    tolerance_group_size = 64
    tolerance_groups = zhmiscellany.list.split_into_sublists(tasks, tolerance_group_size)
    total_groups = len(tolerance_groups)
    
    # processing tasks
    grouped_results = []
    for i, group in enumerate(tolerance_groups):
        # update group with group indexes
        for j, _ in enumerate(group):
            group[j][1][2] = i
        
        # create result mask
        group_files = [task[1][0] for task in group]
        result_files_text = {}
        for file in group_files:
            result_files_text[file] = None
        
        processed_chunk = zhmiscellany.processing.batch_multiprocess(group, expect_crashes=True)
        
        # update mask
        for pair in processed_chunk:
            if pair is not None:
                file = pair[0]
                text = pair[1]
                if isinstance(text, str):
                    result_files_text[file] = text
        
        zhmiscellany.fileio.save_object_to_file(result_files_text, f'{file_name}_{zhmiscellany.string.get_universally_unique_string()}.pkl')
        
        # update final dict
        for file, text in list(result_files_text.items()):
            if not file in files_text:
                files_text[file] = text
            elif files_text[file] is None and text is not None:
                files_text[file] = text

splash.set_progress(100, 100, 'Formatting data...')

print('Formating data')
all_files = []
for file_path, text_content in files_text.items():
    if text_content is not None:
        if text_content != '':
            all_files.append((file_path, text_content))

splash.set_progress(100, 100, 'Creating gui...')

print('Creating GUI')


def search_results_TF_IDF(search, all_files, output_limit):
    # Extract texts for TF-IDF calculation
    texts = [doc[1] for doc in all_files]  # Extract only the text for TF-IDF
    all_texts = [search] + texts
    
    # Compute TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate cosine similarity between search string and documents
    search_vector = tfidf_matrix[0]  # TF-IDF vector for the search string
    document_vectors = tfidf_matrix[1:]  # TF-IDF vectors for the documents
    scores = cosine_similarity(search_vector, document_vectors).flatten()
    
    # Attach scores to the original tuples (metadata, text, score)
    scored_docs = [(doc[0], doc[1], score) for doc, score in zip(all_files, scores)]
    
    # Rank documents by relevance
    ranked_docs = sorted(scored_docs, key=lambda x: x[2], reverse=True)
    
    ranked_docs = [each for each in ranked_docs if each[2] > 0]
    ranked_docs = ranked_docs[:output_limit]
    return ranked_docs


def search_results_fuzzy_search(search, all_files, output_limit):
    results = []
    
    def engine_atom(files):
        for file in files:
            score = fuzz.partial_ratio(search, file[1])
            if score > 70:
                results.append((*file, score))
        return results
    
    num_groups = 32
    
    file_groups = zhmiscellany.list.split_into_n_groups(all_files, num_groups)
    
    tasks = [(engine_atom, (files,)) for files in file_groups]

    results = zhmiscellany.processing.batch_multiprocess(tasks, flatten=True)
    
    results = sorted(results, key=lambda x: x[2], reverse=True)
    results = results[:output_limit]
    return results


def search_engine(text_input):
    global engine_time
    
    def ensure_max_size(img, max_width, max_height):
        """Resize image if it exceeds max dimensions while maintaining aspect ratio."""
        if img.width > max_width or img.height > max_height:
            img.thumbnail((max_width, max_height), Image.LANCZOS)  # Resizes in-place
        return img
    
    zhmiscellany.misc.time_it(None)
    zhmiscellany.misc.time_it(None, 'all')
    
    output_limit = 2**7
    
    #ranked_data = search_results_TF_IDF(text_input, all_files, output_limit)
    ranked_data = search_results_fuzzy_search(text_input, all_files, output_limit)
    
    engine_time = zhmiscellany.misc.time_it('Search engine')
    
    def load_atom(data):
        img = load_image(data[0])
        if img is not None:
            img = ensure_max_size(img, max_image_size, max_image_size)
            img = pil_to_data(img)
            return (img, file_path)
        else:
            return None
    
    tasks = [(load_atom, (data,)) for data in ranked_data]
    
    images_text = zhmiscellany.processing.batch_multiprocess(tasks)
    images_text = [each for each in images_text if each is not None]
    
    image_read = zhmiscellany.misc.time_it('Reading images')
    
    return images_text


# Assume images_text is a list of (PIL_image, "some text") tuples
images_text = []


def pil_to_data(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


app = Flask(__name__)


@app.route('/')
def index():
    html = """
    <html>
    <head>
      <style>
        .container {
          max-height: 97vh;
          overflow-y: scroll;
          display: flex;
          flex-wrap: wrap;
        }
        .item {
          margin: 5px;
          text-align: center;
        }
        img {
          max-width: 100%;
          height: auto;
          display: block;
        }
        .text-container {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
      </style>
      <script>
        // Set text width to match image width after images load
        window.onload = function() {
          const items = document.querySelectorAll('.item');
          items.forEach(item => {
            const img = item.querySelector('img');
            const textContainer = item.querySelector('.text-container');
            // Set text container width to match the actual image width
            textContainer.style.width = img.offsetWidth + 'px';
          });
        }
      </script>
    </head>
    <body>
      <div class="container">
        {% for img, text in items %}
        <div class="item">
          <img src="data:image/png;base64,{{ img }}" alt="Image">
          <div class="text-container" title="{{ text }}">{{ text }}</div>
        </div>
        {% endfor %}
      </div>
    </body>
    </html>
    """
    return render_template_string(html, items=images_text)


import sys
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QSizePolicy
from PyQt5.QtWebEngineWidgets import QWebEngineView

port = random.randint(50000, 64000)

renderer_url = f'http://127.0.0.1:{port}'


class page_renderer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('OCR Search Engine')
        self.setGeometry(100, 100, 1024, 768)
        
        # Create web view
        self.webview = QWebEngineView()
        self.webview.load(QUrl(renderer_url))
        self.webview.loadFinished.connect(self.on_load_finished)
        
        # Create search bar
        self.search_bar = QLineEdit()
        self.default_status = "Search for something!"
        self.status_bar = QLabel(self.default_status, self)
        
        # Fix QLabel Height Issue
        self.status_bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.status_bar.setMaximumHeight(self.search_bar.sizeHint().height())  # Match input field height
        self.status_bar.setAlignment(Qt.AlignVCenter)  # Keep text centered
        
        self.search_bar.returnPressed.connect(self.run_search)
        
        # Button layout
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.search_bar)
        nav_layout.addWidget(self.status_bar)
        
        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(nav_layout)
        layout.addWidget(self.webview)
        
        self.setLayout(layout)
    
    def run_search(self):
        global images_text
        search_text = self.search_bar.text()
        
        self.update_status_bar('Searching...')
        images_text = search_engine(search_text)
        
        self.update_status_bar('Rendering...')
        self.webview.load(QUrl(renderer_url))
    
    def update_status_bar(self, q):
        self.status_bar.setText(q)
        QApplication.processEvents()
    
    def on_load_finished(self):
        zhmiscellany.misc.time_it('Rendering')
        total_time = zhmiscellany.misc.time_it('all', 'all')
        if total_time > 0.0:
            self.update_status_bar(f'{len(images_text)} results in {round(engine_time, 1)}s')
        else:
            self.update_status_bar(self.default_status)


if __name__ == '__main__':
    max_image_size = 2**11
    
    zhmiscellany.processing.start_daemon(target=app.run, kwargs={"port": port})
    app = QApplication(sys.argv)
    renderer = page_renderer()
    renderer.show()
    
    try:
        splash.destroy()
    except RuntimeError:
        pass
    
    sys.exit(app.exec_())