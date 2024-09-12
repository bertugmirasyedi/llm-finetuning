from openai import OpenAI
from typing import Literal


class LLM:
    def __init__(
        self, client: OpenAI, embedding_method: Literal["huggingface", "lmstudio"] = "huggingface"
    ):
        self.client = client
        self.embedding_method = embedding_method

    def generate_questions_and_answers(self, indexed_chunks):
        import random

        responses = list()

        try:
            for chunk in indexed_chunks:
                # Select a random temperature between 0.0 and 0.2
                temperature = round(random.uniform(0.0, 0.2), 2)

                # Define the prompt
                completion = self.client.chat.completions.create(
                    model="bartowski/Llama-3-8B-Instruct-Gradient-1048k-GGUF",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can generate questions and answers from the provided text.",
                        },
                        {
                            "role": "user",
                            "content": f"Generate {self.n_questions} questions and answers from the provided text. Output should only include the questions and answers. There must not be any other text, dash, line, or output.",
                        },
                        {"role": "user", "content": chunk.text},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )

                # Get the response
                response = completion.choices[0].message.content.strip("</s>")

                # Add the response to the list
                responses.append(response)
        except KeyboardInterrupt:
            return responses

        return responses

    def convert_to_json(self, questions_and_answers):
        import json

        responses = list()

        for i in range(len(questions_and_answers)):
            # Define the prompt
            completion = self.client.chat.completions.create(
                model="bartowski/Llama-3-8B-Instruct-Gradient-1048k-GGUF",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that convert the provided text to valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": "Convert the provided text to valid JSON. Output should only be the valid JSON. For example: {[{'question': question1, 'answer': answer1}, {'question': question2, 'answer': answer2}, {'question': question3, 'answer': answer3}]}. There must not be any other text, dash, line, or output.",
                    },
                    {"role": "user", "content": questions_and_answers[i]},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            # Get the response
            response = completion.choices[0].message.content.strip("</s>")

            # Try to convert the response to a JSON object
            # If the response is not a valid JSON object, return an empty list
            try:
                response_json = json.loads(response)
            except json.JSONDecodeError:
                response_json = []

            # Add the response to the list
            responses.append(response_json)

        return responses

    def generate_dataset(self, indexed_chunks):
        response = self.generate_questions_and_answers(indexed_chunks)
        response_json = self.convert_to_json(response)

        return response_json

    def generate_embedding(self, text: str, model_name: str = "bert-base-uncased"):
        if self.embedding_method == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings

            # Load the embedding model
            embeddings = HuggingFaceEmbeddings(model_name=model_name)

            # Generate the embedding
            return embeddings.embed_query(text)

        elif self.embedding_method == "lmstudio":
            pass

    def query_dataset(self, prompt: str, indexed_chunks):
        # Define the prompt
        completion = self.client.chat.completions.create(
            model="bartowski/Llama-3-8B-Instruct-Gradient-1048k-GGUF",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answer the question based on the provided dataset.",
                },
                {
                    "role": "user",
                    "content": f"Dataset: {indexed_chunks[0].text}",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "text"},
        )

        # Get the response
        response = completion.choices[0].message.content.strip("</s>")

        return response
