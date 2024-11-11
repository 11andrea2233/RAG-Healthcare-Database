AI Receptionist Chatbot

This project is an AI Receptionist Chatbot built with Streamlit, FAISS, and OpenAI's API, designed to assist healthcare providers in managing patient inquiries, appointment scheduling, and other administrative tasks in a HIPAA-compliant way. The chatbot combines Retrieval-Augmented Generation (RAG) to deliver accurate and secure responses based on patient information and clinic data.
Features

    Appointment Scheduling: Help patients schedule, reschedule, or cancel appointments based on provider availability.
    Medication Reminders: Provide medication information or refill options if authorized.
    Billing Inquiries: Respond to basic billing questions and direct to billing departments as needed.
    Clinic Information: Share non-sensitive details like clinic hours, location, and contact information.
    Privacy and Security: Ensures patient privacy through identity verification and role-based access.

Technologies Used

    Streamlit: For the web-based user interface.
    FAISS: For efficient similarity search and retrieval.
    OpenAI API: For embedding generation and natural language responses.
    Pandas: For data handling and manipulation.

Project Structure

├── Healthcare_AIchatbot.py               # Main Streamlit app
├── Database.csv         # Example patient dataset
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies

Installation

    Clone the repository:

git clone https://github.com/11andrea2233/RAG-Healthcare-Database.git
cd AI-Receptionist-Chatbot

Install dependencies:

pip install -r requirements.txt

Add your OpenAI API Key:

Set your OpenAI API key as an environment variable:

export OPENAI_API_KEY='your_openai_api_key'

Prepare the Dataset:

    Ensure your patient data is in patients.csv. You can modify this to include real or anonymized data with columns such as Patient ID, Name, DOB, Medications, and Conditions.
    If not available, a sample CSV with mock data should be added to test the application.

Generate and Save Embeddings:

Run the following code snippet to create embeddings for the dataset and save them in a FAISS index for efficient retrieval.

    import pandas as pd
    import openai
    import faiss
    import numpy as np

    # Load the dataset
    df = pd.read_csv('Database.csv')

    # Create embeddings for each row
    def embed_text(text):
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return response['data'][0]['embedding']

    df['embeddings'] = df.apply(lambda row: embed_text(str(row.to_dict())), axis=1)

    # Save embeddings in a FAISS index
    dimension = len(df['embeddings'][0])  # assuming all embeddings have the same dimension
    index = faiss.IndexFlatL2(dimension)
    embeddings = np.array(df['embeddings'].tolist()).astype('float32')
    index.add(embeddings)


Usage

To run the chatbot:

streamlit run app.py

App Interface

    User Query: The patient enters a query (e.g., “I’d like to book an appointment”).
    Identity Verification: The chatbot verifies the patient’s identity before accessing sensitive information.
    Response Generation: The chatbot retrieves relevant data from the FAISS index and generates a response based on a predefined system prompt.
    Output: The chatbot displays the response and offers to assist with any additional questions.

Example Interactions

Appointment Booking:

    User: "I need to schedule a follow-up appointment."
    Chatbot: "Of course! Could you provide your Patient ID or name and DOB for verification?"

Medication Reminder:

    User: "Can you remind me of my medication?"
    Chatbot: "Certainly! Please verify your identity first."

Contributing

Contributions are welcome! To contribute:

    Fork the repository.
    Create a new branch: git checkout -b feature/your-feature
    Commit your changes: git commit -m 'Add some feature'
    Push to the branch: git push origin feature/your-feature
    Submit a pull request.
