import os

if 'RERANKING_CACHE_DIR' not in os.environ:
    os.environ['RERANKING_CACHE_DIR'] = os.path.join(os.path.expanduser('~'),'.cache','rankify')