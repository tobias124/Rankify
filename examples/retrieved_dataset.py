import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from rankify.dataset.dataset import Dataset ,Document, Context, Question,Answer
from rankify.metrics.metrics import Metrics
#Dataset.avaiable_dataset()


datasets = ["nq-test"]#, "ChroniclingAmericaQA-test" , "ArchivialQA-test"]#["nq-dev", "nq-test" , "squad1-test", "trivia-dev", "trivia-test", "webq-test", "squad1-dev" ] #

for name in datasets:
    print("*"*100)
    print(name)
    dataset= Dataset('bm25', name , 1000)
    documents = dataset.download(force_download=False)

    print(len(documents[0].contexts),documents[0].answers )

    metrics = Metrics(documents)

    before_ranking_metrics = metrics.calculate_retrieval_metrics(ks=[1,5,10,20,50,100],use_reordered=False)
    print(before_ranking_metrics)
    print("#"*100)