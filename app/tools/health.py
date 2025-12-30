from app.resources.stats import get_system_stats
from app.utils.logger import logger

def health_check():
    """Check the health status of the RAG system."""
    try:
        stats = get_system_stats()
        return {
            "status": "healthy",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
