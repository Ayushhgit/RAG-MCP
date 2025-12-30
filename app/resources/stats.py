import os
from pathlib import Path
from app.config import INDEX_DIR, PROCESSED_DIR, RAW_DIR
from app.utils.ytils import load_json_file

def get_index_stats():
    """Get statistics about the vector index."""
    index_path = INDEX_DIR / "faiss.index"
    meta_path = INDEX_DIR / "metadata.json"

    stats = {
        "index_exists": index_path.exists(),
        "metadata_exists": meta_path.exists(),
        "index_size_mb": 0,
        "total_documents": 0
    }

    if index_path.exists():
        stats["index_size_mb"] = index_path.stat().st_size / (1024 * 1024)

    if meta_path.exists():
        metadata = load_json_file(meta_path)
        stats["total_documents"] = len(metadata)

    return stats

def get_data_stats():
    """Get statistics about the data directories."""
    raw_files = list(RAW_DIR.glob("*"))
    processed_files = list(PROCESSED_DIR.glob("*"))

    return {
        "raw_files_count": len(raw_files),
        "processed_files_count": len(processed_files),
        "raw_dir_size_mb": sum(f.stat().st_size for f in raw_files) / (1024 * 1024) if raw_files else 0,
        "processed_dir_size_mb": sum(f.stat().st_size for f in processed_files) / (1024 * 1024) if processed_files else 0
    }

def get_system_stats():
    """Get overall system statistics."""
    return {
        "index": get_index_stats(),
        "data": get_data_stats()
    }