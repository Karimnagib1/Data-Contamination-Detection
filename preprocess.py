from datasets import load_dataset, Dataset, DatasetDict



def preprocess_mintaka(train_file, valid_file, test_file):
    train_dataset = load_dataset('parquet', data_files=train_file, split="train")
    valid_dataset = load_dataset('parquet', data_files=valid_file, split="train")
    test_dataset = load_dataset('parquet', data_files=test_file, split="train")

    def format_dataset(dataset):
        data = {"source": [], "messages": [], "num_turns": []}
        for i in range(len(dataset)):
            data["source"].append(dataset[i]["category"])
            message = [{"content": dataset[i]["question"], "role": "user"}, {"content": dataset[i]["answerText"], "role": "assistant"}]
            data["messages"].append(message)
            data["num_turns"].append(len(message))
        new_dataset = Dataset.from_dict(data)
        return new_dataset
    
    train_dataset = format_dataset(train_dataset)
    valid_dataset = format_dataset(valid_dataset)
    test_dataset = format_dataset(test_dataset)

    dataset = DatasetDict({"train": train_dataset, "validation": valid_dataset, "test": test_dataset})

    return dataset

if __name__ == "__main__":
    train_file = "./data/Mintaka/mintaka_train.parquet"
    valid_file = "./data/Mintaka/mintaka_validation.parquet"
    test_file = "./data/Mintaka/mintaka_test.parquet"
    dataset = preprocess_mintaka(train_file, valid_file, test_file)
    dataset.save_to_disk("data/mintaka")


    