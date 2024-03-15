# Introduction
This is a Flask-based web application that allows users to add PDF documents to a Qdrant vector store for searching. The application uses the LlamaIndex library to create embeddings and store the documents. The embeddings are created using the HuggingFace Embedding model. The application also uses the OpenAI API for querying the stored documents.

# Installation
To set up the application, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt` in the terminal.

# Usage
To use the application, follow these steps:

1. Start the application by running `python app.py` in the terminal.
2. The application will be available at `http://localhost:5000/`.

The application has two endpoints:

1. `/add_pdf`: This endpoint allows users to upload a PDF document and store it in the Qdrant vector store.
2. `/query`: This endpoint allows users to query the stored documents using the OpenAI API.

# Contributing
Contributions to this project are welcome. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.



