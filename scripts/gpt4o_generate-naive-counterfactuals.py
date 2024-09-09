import sys
import gc
import re
import tiktoken
import csv
import pandas as pd
import traceback


def count_tokens(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def truncate_text(text, encoding_name='o200k_base', max_tokens=256):
    # Tokenize the text using tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    tokenized_text = encoding.encode(text)
    num_tokens = len(tokenized_text)

    if num_tokens > max_tokens:
        # Calculate the number of tokens to keep
        num_tokens_to_keep = max_tokens - 1  # Subtract 1 to leave space for the truncation token

        # Truncate the tokenized text
        truncated_tokenized_text = {
            'tokens': tokenized_text[:num_tokens_to_keep],
            'is_truncated': True
        }

        # Convert the truncated tokenized text back to a readable string
        truncated_text = encoding.decode(truncated_tokenized_text['tokens'])
        return truncated_text, True
    else:
        return text, False


def generate_set_from_csv(input_string):
    # Split the input string using the comma as the separator
    elements = input_string.split(',')

    # Create a set from the list of elements after stripping whitespaces
    result_set = set(element.strip() for element in elements)

    return result_set

def parse_text_within_tags(input_text):
    # Define the regular expression pattern to match text within "<new>" or "<New>" tags
    pattern = r'<[nN]ew>(.*?)<\/[nN]ew>'
    
    # Find all matches of the pattern in the input text
    matches = re.findall(pattern, input_text)

    if matches == []:
        pattern = r'<[nN]ew>.*?(?=<|$)'
        matches = re.findall(pattern, input_text)

    if matches == []:
        pattern = r'<(.*?)>'
        matches = re.findall(pattern, input_text)
    
    return matches

def generate_comma_separated_string(input_set):
    # Convert the set elements to strings and join them using the comma separator
    result_string = ', '.join(str(element) for element in input_set)

    return result_string


def get_opposite_label(pred_label, task):
    if task=='snli':
        opp_map = {'contradiction':'entailment', 'entailment':'contradiction', 'neutral':'contradiction'}
        return opp_map[pred_label]
    elif task=='imdb':
        ## 0 is negative, 1 is positive
        opp_map = {'positive': 'negative', 'negative':'positive', 0.0:1.0, 1.0:0.0}
        return opp_map[pred_label]

    elif task=='ag_news':
        opp_map = {'the world': 'business', 'business':'sports', 'sports':'the world', 'science/tech': 'sports'}
        return opp_map[pred_label]


def main(args):

    file_path_map = {
        'distilbert-snli': "../data-files/distilbert-snli-triples.csv",
        'distilbert-imdb': "../data-files/distilbert-imdb-triples.csv",
        'distilbert-ag_news': "../data-files/distilbert-ag_news-triples.csv"
    }


    test_type = args.test_type
    task = args.task

    all_data = []
    correctly = []
    with open(file_path_map[test_type], 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)

        # Skip the header (first row)
        next(csv_reader)

        # Iterate over the rows and append data to the list
        for row in csv_reader:
            all_data.append(row)
            if row[1]==row[2]:
                correctly.append(row)

    print("all data: ", len(all_data), ", correctly classified: ", len(correctly), ", accuracy: ", float(len(correctly))/len(all_data))

    MODEL = args.model
    from openai import OpenAI 
    import os

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



    MAX_TOKENS_CONTEXT = 4096

    cf_explanations = []
    noisy_explanations = []
    parsing_fail = 0

    column_names = ["original_text", "ground_truth", "y_pred_original"]
    out_df = pd.DataFrame(correctly[:args.num_samples], columns=column_names)

    task_desc = {
                'snli': "natural language inference on the SNLI dataset",
                'imdb': "sentiment classification on the IMDB dataset",
                'ag_news': "news topic classification on the AG News dataset"
                }

    for i, instance in enumerate(correctly[:args.num_samples]):

        sys.stdout.write("data instance: "+str(i)+"\n")
        sys.stdout.flush()
        text = instance[0]
        gt = instance[1]
        pred = instance[2]

        truncated_text, is_truncated = truncate_text(text)
        text = truncated_text

        opposite_label = get_opposite_label(pred, task=task)

        initial_prompt = "In the task of "+task_desc[task]+", a trained black-box classifier correctly predicted the label '"+pred+" for the following text. Generate a counterfactual explanation from the input text, so that the label changes from '"+pred+"' to '"+opposite_label+"'. Use the following definition of 'counterfactual explanation': \"A counterfactual explanation reveals what should have been different in an instance to observe a diverse outcome.\". Enclose the generated text within \"<new>\" tags.\n---\nInput Text: "+text+"\n---\n"

        
        ## messages should be list of dicts with 'role' and 'content' fields
        messages = [
            {
                'role': 'system',
                'content': 'Follow the instructions as closely as possible. Output exactly in the format that is specified by the user.'
            },
            {
            'role':'user',
            'content':initial_prompt
        }]

        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages
            )

        cf_response = completion.choices[0].message.content
        
        gc.collect()
        
        if parse_text_within_tags(cf_response)==[]:
            cf_explanations.append('null')
            noisy_explanations.append(cf_response)
            parsing_fail+=1
        else:
            cf_explanations.append(parse_text_within_tags(cf_response)[0].strip())
            noisy_explanations.append('null')
        


    print('Done!')
    print('Parsing Fail Count: ', parsing_fail)
    try:
        out_df['counterfactual_text'] = cf_explanations
        out_df['noisy_cf'] = noisy_explanations
        


        out_df.to_csv(f"no_spec-{MODEL}-" + test_type + "-cf-explanations.csv", index=False)
    except Exception as e:
        print(cf_explanations)
        print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate counterfactual explanations using GPT4o or GPT4o-mini  family of models")
    parser.add_argument("--model", type=str, default="gpt-4o", choices=['gpt-4o', 'gpt-4o-mini'], help="The model to use for generating explanations")
    parser.add_argument("--task", type=str, default="snli", choices=['snli', 'ag_news', 'imdb'], help="The task for which the explanations are generated")
    parser.add_argument("--test_type", type=str, default="distilbert-snli", help="The type of test data to use for generating explanations, should match task")
    parser.add_argument("--num_samples", type=int, default=500, help="The number of samples to process")
    args = parser.parse_args()
    main(args)
