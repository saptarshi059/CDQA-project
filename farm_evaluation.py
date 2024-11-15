from farm.utils import initialize_device_settings
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import TextClassificationProcessor, SquadProcessor
from farm.data_handler.data_silo import DataSilo
from farm.eval import Evaluator
from farm.modeling.adaptive_model import AdaptiveModel
from pathlib import Path


def evaluate_question_answering():
    ##########################
    ########## Settings
    ##########################
    device, n_gpu = initialize_device_settings(use_cuda=True)
    # lang_model = "deepset/roberta-base-squad2-covid"
    lang_model = "deepset/roberta-base-squad2"
    # deepset/roberta-base-squad2

    do_lower_case = True

    data_dir = Path('data/')
    # evaluation_filename = 'COVID-QA_cleaned.json'
    evaluation_filename = '200423_covidQA.json'

    batch_size = 128
    no_ans_boost = -100
    accuracy_at = 3     # accuracy at n is useful for answers inside long documents

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=384,
        label_list= ["start_token", "end_token"],
        metric="squad",
        train_filename=None,
        dev_filename=None,
        dev_split=0,
        test_filename=evaluation_filename,
        data_dir=data_dir,
        doc_stride=192,
    )

    # 3. Create a DataSilo that loads dataset, provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # 4. Create an Evaluator
    evaluator = Evaluator(
        data_loader=data_silo.get_data_loader("test"),
        tasks=data_silo.processor.tasks,
        device=device
    )

    # 5. Load model
    model = AdaptiveModel.convert_from_transformers(lang_model, device=device, task_type="question_answering")
    # use "load" if you want to use a local model that was trained with FARM
    # model = AdaptiveModel.load(lang_model, device=device)
    model.prediction_heads[0].no_ans_boost = no_ans_boost
    model.prediction_heads[0].n_best = accuracy_at
    model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)

    # 6. Run the Evaluator
    results = evaluator.eval(model)
    f1_score = results[0]["f1"]
    em_score = results[0]["EM"]
    tnacc = results[0]["top_n_accuracy"]
    print("F1-Score:", f1_score)
    print("Exact Match Score:", em_score)
    print(f"top_{accuracy_at}_accuracy:", tnacc)


if __name__ == "__main__":
    evaluate_question_answering()
