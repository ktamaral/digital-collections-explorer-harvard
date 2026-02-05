# Digital Collections Explorer - Technical Introduction

## Resources

- **Main Repository**: https://github.com/hinxcode/digital-collections-explorer
- **Research Paper**: https://arxiv.org/abs/2507.00961
- **GovScape**: https://www.govscape.net/ | https://arxiv.org/abs/2511.11010
- **CLIP Paper**: https://arxiv.org/abs/2103.00020
- **Hugging Face CLIP**: https://huggingface.co/docs/transformers/model_doc/clip

## System Overview

Digital Collections Explorer is an **open-source, plug-and-play multimodal search platform** that enables both text-based and image-based searches across visual collections using **CLIP (Contrastive Language-Image Pre-training)** embeddings. It's designed to democratize access to digital archives, particularly those with limited metadata.

### Key Papers

- **Digital Collections Explorer**: [https://arxiv.org/abs/2507.00961](https://arxiv.org/abs/2507.00961)
- **GovScape** (production deployment using this platform): [https://arxiv.org/abs/2511.11010](https://arxiv.org/abs/2511.11010)

---

## Core Architecture

### 1. Embedding Generation Pipeline

**File**: [src/models/clip/generate_embeddings.py](src/models/clip/generate_embeddings.py)

The embedding pipeline processes visual content and generates vector representations:

- **Input Sources**:
  - Images: JPG, JPEG, PNG, GIF, BMP, TIFF, WebP
  - PDFs: Each page converted to an image and processed separately
- **Processing Steps**:
  1. Recursively scans `data/raw` directory for supported files
  2. For each image/PDF page:
     - Creates **thumbnail** (400x400px, JPEG quality 80)
     - Creates **processed version** (max 1920x1920px, JPEG quality 90)
     - Generates **CLIP embedding** using the configured model
  3. Saves outputs to organized directories

- **Output Files**:
  - `data/embeddings/embeddings.pt` - Stacked PyTorch tensor of all embeddings
  - `data/embeddings/item_ids.pt` - List of unique document identifiers
  - `data/embeddings/metadata.json` - File paths and metadata for each item

- **PDF Handling**:
  - Each page becomes a separate searchable item
  - Item IDs format: `{base64_encoded_path}_{page_number}`
  - Preserves page count and relationships in metadata

### 2. Backend API (FastAPI + Python)

**File**: [src/backend/main.py](src/backend/main.py)

The backend provides a RESTful API for searching and serving images.

#### Core Services

**CLIPService** ([src/backend/services/clip_service.py](src/backend/services/clip_service.py))

- Loads the CLIP model (default: `openai/clip-vit-base-patch32`)
- Encodes text queries into embeddings
- Encodes uploaded images into embeddings
- Auto-detects CUDA availability, falls back to CPU

**EmbeddingService** ([src/backend/services/embedding_service.py](src/backend/services/embedding_service.py))

- Loads pre-computed embeddings into memory at startup
- Performs vector similarity search using cosine similarity
- Handles pagination (limit + offset)
- Retrieves metadata for search results

#### API Endpoints

**Search Endpoints** ([src/backend/api/routes/search.py](src/backend/api/routes/search.py))

```
GET /api/search/text?query={text}&limit={n}&page={p}
```

- Takes natural language text query
- Encodes to embedding using CLIP
- Uses logit_scale parameter for calibrated similarity scores
- Returns ranked results with scores and metadata

```
POST /api/search/image
```

- Accepts uploaded image file
- Encodes image to embedding using CLIP
- Returns similar images from collection
- Supports pagination

**Image Serving** ([src/backend/api/routes/images.py](src/backend/api/routes/images.py))

```
GET /images/{id}?size={thumbnail|full}
```

- Serves optimized thumbnails or full processed images
- Looks up paths from metadata using document ID

```
GET /static/{id}
```

- Serves original source files
- Provides download with proper filename

**Health & Stats**

```
GET /api/health
```

- Health check endpoint

```
GET /api/embeddings/stats
```

- Returns total count of indexed items

#### Search Mechanism

The core search algorithm uses cosine similarity:

```python
similarities = torch.matmul(embeddings, query_embedding.t()).squeeze()
```

Since embeddings are L2-normalized, this dot product equals cosine similarity. The system:

1. Computes similarity scores for all items
2. Applies optional logit_scale (for text queries)
3. Finds top-k results using `torch.topk`
4. Applies pagination offset and limit
5. Returns results with scores and metadata

### 3. Frontend Variants (React + Vite)

Three specialized React applications share the same backend API:

#### Photographs ([src/frontend/photographs](src/frontend/photographs))

- Photo gallery interface
- Grid layout with lightbox viewing
- Image upload for reverse image search
- Optimized for photographic collections

#### Maps ([src/frontend/maps](src/frontend/maps))

- Map-specific viewer
- IIIF integration for Library of Congress collections
- Supports high-resolution map viewing
- Metadata display for cartographic information

#### Documents ([src/frontend/documents](src/frontend/documents))

- PDF page search interface
- Card-based results showing page previews
- Designed for born-digital documents and web archives
- Page-level granularity

#### Shared Features

All three frontends provide:

- **Text search**: Natural language queries
- **Image search**: Upload similar images to find matches
- **Pagination**: Navigate through large result sets
- **Responsive design**: Works on desktop and mobile
- **Real-time results**: Instant search feedback

---

## Configuration & Setup

### Setup Process

The project uses a Node.js setup script to configure collection type and build the frontend:

```bash
npm run setup -- --type=photographs
```

**What it does** ([src/setup.js](src/setup.js)):

1. Updates `config.json` with selected collection type
2. Navigates to the appropriate frontend directory
3. Installs npm dependencies
4. Builds the frontend using Vite
5. Places build artifacts in `src/frontend/{type}/dist`

### Configuration File

**File**: [config.json](config.json)

```json
{
  "collection_type": "photographs",
  "raw_data_dir": "data/raw",
  "processed_data_dir": "data/processed",
  "embeddings_dir": "data/embeddings",
  "thumbnails_dir": "data/thumbnails",
  "model_config": {
    "clip_model": "openai/clip-vit-base-patch32",
    "batch_size": 32,
    "device": "cuda"
  },
  "api_config": {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": false
  }
}
```

**Configuration Options**:

- `collection_type`: Which frontend to serve (`photographs`, `maps`, `documents`)
- `raw_data_dir`: Source directory for images/PDFs to index
- `embeddings_dir`: Where to store generated embeddings
- `clip_model`: Hugging Face model identifier (can use custom models)
- `device`: `cuda` for GPU, `cpu` for CPU-only
- `batch_size`: Number of images to process simultaneously
- `debug`: Enable hot-reloading for development

### Directory Structure

```
data/
├── raw/              # Place your images/PDFs here
├── processed/        # Auto-generated: optimized images
├── thumbnails/       # Auto-generated: preview thumbnails
└── embeddings/       # Auto-generated: CLIP vectors
    ├── embeddings.pt
    ├── item_ids.pt
    └── metadata.json
```

---

## Deployment

### Standard Deployment

1. **Prepare collection**: Place images in `data/raw/`
2. **Generate embeddings**: `python -m src.models.clip.generate_embeddings`
3. **Start server**: `python -m src.backend.main`
4. **Access**: Navigate to `http://localhost:8000`

### Docker Deployment

**File**: [Dockerfile](Dockerfile)

Multi-stage build process:

1. **Stage 1** (Node.js): Builds frontend
2. **Stage 2** (Python): Sets up runtime environment

**Build with collection type**:

```bash
docker build --build-arg COLLECTION_TYPE=photographs -t collections-explorer .
```

**Optional: Skip embedding generation**:

```bash
docker build --build-arg SKIP_GEN_EMBEDDING=true -t collections-explorer .
```

**Run**:

```bash
docker run -p 8000:8000 -v $(pwd)/data:/app/data collections-explorer
```

### Development Mode

For frontend development with hot-reloading:

1. Set `api_config.debug: true` in `config.json`
2. Start backend: `python -m src.backend.main`
3. Start frontend dev server:
   ```bash
   cd src/frontend/[photographs|maps|documents]
   npm run dev
   ```
4. Access frontend at `http://localhost:5173`

The dev server proxies API requests to the backend.

---

## Key Features & Capabilities

### ✅ Multimodal Search

- **Text queries**: Natural language descriptions ("red barn in winter")
- **Reverse image search**: Upload similar images to find matches
- Works because CLIP embeddings place similar concepts close in vector space

### ✅ Scalable

- Demonstrated on **hundreds of thousands of images** (MacBook Pro M4)
- GovScape deployment: **10M+ PDFs**, **70M+ pages**
- In-memory embedding search is extremely fast
- All embeddings loaded at startup for instant results

### ✅ Flexible Content Support

- **Photographs**: Standard image formats
- **Maps**: High-resolution cartographic materials
- **Documents**: PDF pages become individually searchable
- **Mixed collections**: Can handle multiple content types

### ✅ Low Barrier to Entry

- **No manual tagging required**: CLIP learns visual concepts automatically
- **Minimal setup**: Just place files and run generation script
- **Open source**: Free to use and customize
- **Standard hardware**: Runs on laptops (CPU mode available)

### ✅ Cost Efficient

- **GovScape report**: ~$1,500 compute for 10M PDFs
- **~47,000 PDF pages per dollar** spent on compute
- One-time embedding generation cost
- Search is free after initial processing

### ✅ Customizable

- **Swap CLIP models**: Use any Hugging Face CLIP variant
- **Fine-tune models**: Train on domain-specific data
- **Adjust batch sizes**: Optimize for your hardware
- **Configure paths**: Flexible directory structure
- **Extend APIs**: FastAPI makes adding endpoints easy

---

## Real-World Applications

### GovScape (https://www.govscape.net/)

Production deployment using Digital Collections Explorer codebase:

- **Scale**: 10,015,993 federal government PDFs (70,958,487 total pages)
- **Source**: 2020 End of Term Web Archive
- **Capabilities**:
  - Semantic text search across all pages
  - Visual search (e.g., "redacted documents", "pie charts")
  - Metadata facet filtering (domain, crawl date)
  - Exact text search against extracted content
- **Cost**: ~$1,500 total preprocessing for 10M PDFs
- **Performance**: Demonstrates scalability of architecture

### Other Use Cases

- **Cultural heritage institutions**: Museums, libraries, archives
- **Research collections**: Historical photographs, manuscripts
- **Government transparency**: FOIA documents, public records
- **Web archives**: Screenshots, archived pages
- **Corporate collections**: Product photos, design assets
- **Personal collections**: Organize large photo libraries

---

## Technical Highlights

### L2-Normalized Embeddings

```python
embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
```

- All vectors normalized to unit length
- Enables cosine similarity via simple dot product
- Mathematically: `cos(θ) = dot(a, b) / (||a|| * ||b||) = dot(a, b)` when normalized

### Batch Processing

- Configurable batch size for embedding generation
- Balances GPU memory usage with processing speed
- Default: 32 images per batch

### In-Memory Search

- All embeddings loaded into RAM at startup
- Sub-second search across large collections
- Trade-off: Memory usage scales with collection size
- Typical: ~512 dimensions × 4 bytes × item count

### Base64 URL-Safe ID Encoding

```python
item_id = base64.urlsafe_b64encode(relative_path.encode('utf-8')).decode('utf-8').rstrip('=')
```

- Encodes file paths as URL-safe identifiers
- Preserves file system hierarchy in IDs
- PDF pages: `{base64_path}_{page_num}`

### Static File Serving

- Single FastAPI application serves both:
  - RESTful API endpoints (`/api/*`)
  - Built React frontend (`/`)
- Simplified deployment (one process)
- Configured via `StaticFiles` mount

### CORS Configuration

Pre-configured origins for development:

```python
cors_origins = [
    "http://localhost:8000",
    "http://localhost:5173",  # Vite dev server
]
```

---

## Dependencies

### Backend (Python)

**Core ML**:

- `torch` - PyTorch for tensor operations
- `transformers` - Hugging Face CLIP models
- `open_clip_torch` - Extended CLIP model support

**Web Framework**:

- `fastapi` - Modern async web framework
- `uvicorn` - ASGI server
- `python-multipart` - File upload support

**Image Processing**:

- `pillow` - Image manipulation
- `pdf2image` - PDF to image conversion
- `PyPDF2` - PDF metadata extraction

**Configuration**:

- `pydantic` - Data validation
- `pydantic-settings` - Configuration management

### Frontend (Node.js)

**Framework**:

- `react` - UI library
- `vite` - Build tool and dev server

**Each frontend has specialized dependencies**:

- Document viewer: PDF.js for rendering
- Map viewer: IIIF support libraries
- Photo viewer: Lightbox components

## Extension Points

### Custom CLIP Models

Replace `clip_model` in config.json:

```json
"clip_model": "openai/clip-vit-large-patch14"
```

Or use fine-tuned models:

```json
"clip_model": "path/to/your/finetuned/model"
```

### Additional Metadata

Extend `generate_embeddings.py` to extract:

- EXIF data from photos
- Geographic coordinates
- Timestamps
- Custom taxonomies

### Filtering & Facets

Add to `EmbeddingService.search()`:

- Filter by metadata before ranking
- Add faceted search (by date, location, etc.)
- Combine keyword and semantic search

### Alternative Frontends

Create new frontend variants:

1. Add directory: `src/frontend/my-collection/`
2. Implement React app with search API calls
3. Add to `COLLECTION_TYPES` in `setup.js`
4. Build and serve like existing frontends

### API Extensions

FastAPI makes it easy to add:

- Feedback collection endpoints
- User authentication
- Collection management APIs
- Batch search operations
- Export functionality

## Troubleshooting

### ModuleNotFoundError: No module named '\_lzma'

**Error Message:**

```
ModuleNotFoundError: No module named '_lzma'
```

**Full Stack Trace:**

```python
Traceback (most recent call last):
  File "/path/to/venv/lib/python3.12/site-packages/torchvision/datasets/utils.py", line 4, in <module>
    import lzma
  File "/Users/username/.pyenv/versions/3.12.9/lib/python3.12/lzma.py", line 27, in <module>
    from _lzma import *
ModuleNotFoundError: No module named '_lzma'
```

**Cause:**

This error occurs when Python was compiled without the XZ/LZMA compression library. This is common when:

- Python was installed via pyenv before the `xz` library was available on your system
- Python was compiled from source without XZ development headers

**Solution (macOS with pyenv):**

1. **Install the xz library:**

   ```bash
   brew install xz
   ```

2. **Reinstall Python with pyenv:**

   ```bash
   pyenv uninstall -f 3.12.9
   pyenv install 3.12.9
   ```

   This reinstalls Python and compiles it with xz support.

3. **Recreate your virtual environment:**

   ```bash
   rm -rf venv
   python -m venv venv
   ```

4. **Reinstall dependencies:**

   ```bash
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Verify the fix:**
   ```bash
   python -c "import _lzma; print('✅ _lzma module is working!')"
   ```

**Solution (Linux with pyenv):**

1. **Install XZ development libraries:**

   ```bash
   # Ubuntu/Debian
   sudo apt-get install liblzma-dev

   # Fedora/RHEL
   sudo dnf install xz-devel
   ```

2. **Follow steps 2-5 from macOS solution above**

**Alternative Solution (System Python):**

If you're using system Python (not pyenv), ensure the xz library is installed and reinstall Python:

```bash
# macOS
brew install xz
brew reinstall python@3.12

# Linux (Ubuntu/Debian)
sudo apt-get install liblzma-dev python3-lzma
```

### AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'

**Error Message:**

```
ERROR - Error generating embeddings: 'BaseModelOutputWithPooling' object has no attribute 'norm'
RuntimeError: stack expects a non-empty TensorList
```

**Full Error Log:**

```
2026-02-05 12:48:02,175 - ERROR - Error generating embeddings: 'BaseModelOutputWithPooling' object has no attribute 'norm'
...
Traceback (most recent call last):
  File "/path/to/src/models/clip/generate_embeddings.py", line 342, in main
    embeddings_tensor = torch.stack(list(result.embeddings.values()))
RuntimeError: stack expects a non-empty TensorList
```

**Root Cause:**

This is a **breaking change in transformers v5.0+**. The CLIP model's `get_image_features()` and `get_text_features()` methods changed their return type:

- **transformers v4.x**: Returns `torch.Tensor` directly
- **transformers v5.x**: Returns `BaseModelOutputWithPooling` object

The original code assumed a tensor and called `.norm()` directly, which fails on the ModelOutput object.

**References:**

- [Hugging Face transformers CLIP documentation](https://huggingface.co/docs/transformers/model_doc/clip)
- [BaseModelOutputWithPooling API](https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling)
- [Original code in main branch](https://github.com/hinxcode/digital-collections-explorer/blob/main/src/backend/services/clip_service.py#L34-L54)

**Original Code (from main branch):**

```python
def encode_text(self, texts) -> torch.Tensor:
    """Encode text to embedding"""
    text_inputs = self.processor(text=texts, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

    with torch.no_grad():
        text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu()
```

This code works with transformers v4.x but fails with v5.x.

**Backwards-Compatible Solution:**

Update **both** [src/models/clip/generate_embeddings.py](src/models/clip/generate_embeddings.py) and [src/backend/services/clip_service.py](src/backend/services/clip_service.py):

```python
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

    # Handle both tensor and ModelOutput types (backwards compatible)
    if isinstance(image_features, torch.Tensor):
        # transformers v4.x returns tensor directly
        embeddings = image_features
    elif hasattr(image_features, 'pooler_output'):
        # transformers v5.x prefers pooler_output (2D: [batch, embedding_dim])
        embeddings = image_features.pooler_output
    elif hasattr(image_features, 'last_hidden_state'):
        # Fallback: pool the sequence dimension if only last_hidden_state available
        embeddings = image_features.last_hidden_state[:, 0, :]
    else:
        embeddings = torch.tensor(image_features)

    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
```

**Why This is Backwards Compatible:**

1. **transformers v4.x**: `isinstance(image_features, torch.Tensor)` returns `True` → uses tensor directly ✅
2. **transformers v5.x**: Returns object with `pooler_output` attribute → extracts correct 2D tensor ✅

### RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D

**Error Message:**

```
RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D
```

**Full Error Log:**

```
2026-02-05 13:07:48,728 - src.backend.services.embedding_service - ERROR - Error in search: t() expects a tensor with <= 2 dimensions, but self is 3D
2026-02-05 13:07:48,728 - src.backend.services.embedding_service - ERROR - Traceback (most recent call last):
  File "/Volumes/dev/HUIT/digital-collections-explorer-harvard/src/backend/services/embedding_service.py", line 111, in search
    similarities = torch.matmul(self.embeddings, query_embedding.t()).squeeze()
                                                 ^^^^^^^^^^^^^^^^^^^
RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D
```

**Root Cause:**

This error occurs when using an **incorrect fix** for the BaseModelOutputWithPooling issue above. The problem happens when code checks for `last_hidden_state` **before** `pooler_output`:

**Incorrect fix (causes this error):**

```python
# WRONG ORDER - checks last_hidden_state first
if hasattr(image_features, 'last_hidden_state'):
    embeddings = image_features.last_hidden_state  # 3D tensor!
elif hasattr(image_features, 'pooler_output'):
    embeddings = image_features.pooler_output
```

**Why this fails:**

Understanding the ModelOutput structure (transformers v5.x):

- `pooler_output`: shape `[batch_size, embedding_dim]` → **2D** ✅ (what we want)
- `last_hidden_state`: shape `[batch_size, sequence_length, embedding_dim]` → **3D** ❌ (causes error)

When you extract `last_hidden_state` first:

1. Text: `[1, 3, 512]` (3D)
2. Image: `[1, 50, 768]` (3D)

The search function tries to transpose with `.t()`, which only works on 1D and 2D tensors.

**Test to verify the issue:**

```python
from transformers import CLIPModel, CLIPProcessor
import torch

model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

text_inputs = processor(text=['test'], return_tensors='pt')
result = model.get_text_features(**text_inputs)

print(f"Type: {type(result)}")  # BaseModelOutputWithPooling
print(f"pooler_output shape: {result.pooler_output.shape}")      # [1, 512] ✅
print(f"last_hidden_state shape: {result.last_hidden_state.shape}")  # [1, 3, 512] ❌
```

**Correct Solution:**

Check attributes in the **correct order** (tensor → pooler_output → last_hidden_state):

```python
with torch.no_grad():
    text_features = self.model.get_text_features(**text_inputs)

    # Check in correct order
    if isinstance(text_features, torch.Tensor):
        # v4.x compatibility
        embeddings = text_features
    elif hasattr(text_features, 'pooler_output'):
        # v5.x: Use the 2D pooled embeddings
        embeddings = text_features.pooler_output
    elif hasattr(text_features, 'last_hidden_state'):
        # Fallback only: pool to 2D by taking first token
        embeddings = text_features.last_hidden_state[:, 0, :]
    else:
        embeddings = torch.tensor(text_features)

    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
```

**Files requiring this fix:**

- [src/backend/services/clip_service.py](src/backend/services/clip_service.py) - Both `encode_text()` and `encode_image()`
- [src/models/clip/generate_embeddings.py](src/models/clip/generate_embeddings.py) - `generate_embeddings()` function

**After updating, restart the backend:**

```bash
# If running locally:
python -m src.backend.main

# If using Docker:
docker compose restart backend
```

**Summary Table:**

| transformers             | get_text_features() returns                    | Shape              | Works with .t()?     |
| ------------------------ | ---------------------------------------------- | ------------------ | -------------------- |
| v4.x                     | `torch.Tensor`                                 | `[1, 512]` (2D)    | ✅ Yes               |
| v5.x (pooler_output)     | `BaseModelOutputWithPooling.pooler_output`     | `[1, 512]` (2D)    | ✅ Yes               |
| v5.x (last_hidden_state) | `BaseModelOutputWithPooling.last_hidden_state` | `[1, 3, 512]` (3D) | ❌ No - causes error |

**Backwards compatibility** The fix checks for tensor type first (v4.x), then pooler_output (v5.x), then falls back to pooling last_hidden_state if needed.

# Goals for Harvard LTS contributions

1. Create a docker compose file and instructions to orchestrate a stack of individual services in a microservices architecture

- Write these updates cleanly, avoiding modifications to existing code when possible
- Upgrade to latest versions of python and transformers package with backwards compatibility if possible
  - Look more into required code updates and make backwards compatible if possible to fix errors such as `'BaseModelOutputWithPooling' object has no attribute 'norm'`
  - Troubleshooting for CLIP model install status unexpected, maybe can be ignored but needs confirmation

    ```
    CLIPModel LOAD REPORT from: openai/clip-vit-base-patch32
    Key                                  | Status     |  |
    -------------------------------------+------------+--+-
    vision_model.embeddings.position_ids | UNEXPECTED |  |
    text_model.embeddings.position_ids   | UNEXPECTED |  |

    Notes:
    - UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    ```

TODO:

- Update requirements.txt with latest package versions
- Test backwards compatibility with older transformers 4.4 package
- Start working on docker compose
- Embedding model evaluation (nice to have but this model seems ok, while not perfect)
- Research how to use multiple embedding models (e.g. GovScape uses both CLIP and BGE)
  - Consider reranking strategies

Questions for UX team:

- Considerations for interface interactions
  - Possible to have multiple tabs similar to Google search:
    - Text mode (current)
    - Image mode (image search)
    - Agent mode (chatbot)
  - Easier from a technical standpoint to have separate modes
    - Can look into a combined search experience, although may be better to do separate interfaces at first and then work on the integrated search as an enhancement, in the interest of developing iteratively as Stu advocates

Known issues:

- Looks like thumbnails are rotated when the images are in portrait view
