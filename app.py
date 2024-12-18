from google.cloud import storage
from dotenv import load_dotenv
import requests
from langchain_community.vectorstores import Chroma
# import yaml
import urllib.parse
import os
import json
from pdf2image import convert_from_path
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import os
import re
import csv
import shutil
import time
from langchain_community.document_loaders import JSONLoader
# from langchain_chroma import Chroma  # Updated import for Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import Tool, AgentType, initialize_agent
import streamlit as st

load_dotenv()
# Write the Google Cloud credentials to a temporary file
with open("google_credentials.json", "w") as f:
    f.write(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_2"])

# Set the environment variable for Google credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_credentials.json"
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
storage_client = storage.Client()
bucket_screenshots = storage_client.bucket('dd-media-files')

def process_pdfs_in_folder():
    """
    Processes all PDFs in the specified folder. 
    For each PDF, creates a separate folder, converts each page to images, and saves the images.
    """
    # Input folder where PDFs are stored
    pdf_folder = "UCB PP"

    # Output folder to store images
    output_base_folder = "Images"

    # Ensure output base folder exists
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
    
    # Iterate through all files in the specified folder
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):  # Process only PDF files
            pdf_path = os.path.join(pdf_folder, filename)
            pdf_name = os.path.splitext(filename)[0]  # PDF name without extension
            
            # Create a folder for this PDF
            output_folder = os.path.join(output_base_folder, pdf_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            print(f"Processing '{filename}'...")
            
            # Convert PDF to images
            try:
                images = convert_from_path(pdf_path)
                for page_number, image in enumerate(images):
                    image_name = f"page_{page_number + 1}.jpg"  # Name for each image
                    image_path = os.path.join(output_folder, image_name)
                    image.save(image_path, "JPEG")  # Save image in JPEG format
                    print(f"Saved: {image_path}")
            except Exception as e:
                print(f"Error processing '{filename}': {e}")
    
    print("\nProcessing complete!")

# # Call the function
# process_pdfs_in_folder()


def upload_to_gcs(image_path, bucket_name):
    """
    Uploads an image to Google Cloud Storage under a specific folder and makes it public.
    Returns the public URL of the image.
    """
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Create the path in GCS: folder/document-comparison/{image_name}
    folder_name = "document-comparison"
    blob_name = f"{folder_name}/{os.path.basename(image_path)}"  # Append image file name to the folder path
    blob = bucket.blob(blob_name)
    
    # Upload the file
    blob.upload_from_filename(image_path)
    blob.make_public()  # Make the file publicly accessible
    
    public_url = blob.public_url
    print(f"Uploaded {image_path} to {public_url}")
    return public_url

# Function to convert an image to HTML using OpenAI's API
def image_to_html(image_path):
    """
    Upload the image to GCS, get a public URL, and convert it to HTML using OpenAI's API.
    """
    # OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is not set. Set it in the environment variable OPENAI_API_KEY.")

    # GCS Bucket name
    bucket_name = 'dd-media-files'  

    # Upload the image to GCS and get its public URL
    image_url = upload_to_gcs(image_path, bucket_name)

    # OpenAI prompt
    prompt = """
    Extract the content from the image while preserving the document's layout, headings, subheadings, paragraphs, 
    and formatting. Include details such as:
    - Bold, italic, underline text styles
    - Font size
    - Font color
    - Highlights or shades
    
    Output clean, semantic HTML with proper tags: <h1>, <h2>, <p>, <strong>, <em>, <u>, <span> with inline styles.
    Ensure the content looks visually similar to the original document.
    """

    # OpenAI API headers and payload
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        "max_tokens": 2000
    }

    # Make the API request
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()
        
        # Extract and return HTML content
        if 'choices' in response_json:
            return response_json['choices'][0]['message']['content']
        else:
            print(f"Error: {response_json}")
            return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Function to process all images in the folder structure and save HTML files
def process_images_to_htmls():
    """
    Access each image folder and convert images to HTML files.
    """
        # Paths
    image_base_folder = "Images"  
    html_base_folder = "HTMLs"    

    os.makedirs(html_base_folder, exist_ok=True)  # Ensure HTML output folder exists
    
    # Traverse through each PDF folder in the images folder
    for pdf_name in os.listdir(image_base_folder):
        pdf_image_folder = os.path.join(image_base_folder, pdf_name)
        if os.path.isdir(pdf_image_folder):  # Check if it's a folder
            print(f"Processing images in folder: {pdf_image_folder}")
            
            # Create corresponding HTML folder
            html_folder = os.path.join(html_base_folder, pdf_name)
            os.makedirs(html_folder, exist_ok=True)
            
            # Process each image in the folder
            for image_filename in os.listdir(pdf_image_folder):
                if image_filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(pdf_image_folder, image_filename)
                    print(f"Converting {image_path} to HTML...")
                    
                    # Convert image to HTML
                    try:
                        html_content = image_to_html(image_path)
                        # Save HTML file
                        html_filename = os.path.splitext(image_filename)[0] + ".html"
                        html_file_path = os.path.join(html_folder, html_filename)
                        
                        with open(html_file_path, "w", encoding="utf-8") as html_file:
                            html_file.write(html_content)
                            print(f"Saved HTML: {html_file_path}")
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

    print("\nAll images have been processed into HTML files.")

def combine_html_files():
    """
    Combine all HTML files for each PDF into a single HTML file.
    """
    html_base_folder='HTMLs'
    # Output folder for combined HTML files
    combined_output_folder = "Combined_HTMLs"
    os.makedirs(combined_output_folder, exist_ok=True)

    # Traverse through each PDF folder in the HTML base folder
    for pdf_name in os.listdir(html_base_folder):
        pdf_html_folder = os.path.join(html_base_folder, pdf_name)

        if os.path.isdir(pdf_html_folder):  # Ensure it's a folder
            print(f"Combining HTML files in folder: {pdf_html_folder}")

            # Combined HTML file path
            combined_html_path = os.path.join(combined_output_folder, f"{pdf_name}.html")

            # Start writing the combined HTML file
            with open(combined_html_path, "w", encoding="utf-8") as combined_file:
                # Write basic HTML structure
                combined_file.write(f"<!DOCTYPE html>\n<html>\n<head>\n<title>{pdf_name}</title>\n</head>\n<body>\n")
                combined_file.write(f"<h1>{pdf_name}</h1>\n")  # Add title

                # Traverse all HTML files in the PDF's folder
                for html_filename in sorted(os.listdir(pdf_html_folder)):
                    if html_filename.endswith(".html"):
                        html_file_path = os.path.join(pdf_html_folder, html_filename)
                        print(f"Adding {html_file_path} to {combined_html_path}")
                        
                        # Read and append the content of each HTML file
                        with open(html_file_path, "r", encoding="utf-8") as html_file:
                            combined_file.write(f"\n<!-- {html_filename} -->\n")
                            combined_file.write(html_file.read())
                
                # Close the HTML structure
                combined_file.write("\n</body>\n</html>")

            print(f"Combined HTML created: {combined_html_path}")

    print("\nAll PDFs' HTML files have been combined successfully.")

def html_to_json(html_content):
    """
    Convert HTML content into structured JSON using an LLM model with validation.
    """
    # Initialize OpenAI GPT-4o-mini model
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Prompt template for extracting JSON from HTML
    prompt_template = f"""
    You are tasked with analyzing an HTML document and transforming it into a structured JSON format.
    Only provide valid JSON, and do not include any explanation, notes, or surrounding code fences.

    ### Instructions:
    1. Extract headings, subheadings, paragraphs, and lists from the provided HTML content.
    2. Organize the content hierarchically into JSON format:
       - Use 'heading' for major headings.
       - Use 'subheading' for subsections.
       - Use 'paragraphs' for content under headings/subheadings.
       - Use 'list' for bullet points.
    3. Include any contact information like emails or addresses under a 'contact_information' key.

    ### HTML Input:
    {html_content}

    Provide only valid JSON output. Do not include anything else.
    """

    try:
        # Invoke LLM
        response = llm.invoke(prompt_template)
        raw_output = response.content.strip()

        # Clean response to remove code fences or extra content
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]  # Remove starting ```json
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]  # Remove ending ```

        # Parse the cleaned response as JSON
        try:
            json_content = json.loads(raw_output)
            return json_content
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}. Raw output saved for debugging.")
            return {"error": "Invalid JSON", "raw_output": raw_output}

    except Exception as e:
        print(f"Error invoking LLM: {e}")
        return {"error": "Exception during LLM processing", "details": str(e)}

def process_combined_html_to_json():
    """
    Process combined HTML files, convert them into JSON, and save the results.
    """
    combined_html_folder = "Combined_HTMLs"
    json_output_folder = "json_responses"
    os.makedirs(json_output_folder, exist_ok=True)

    # Traverse combined HTML files
    for html_filename in os.listdir(combined_html_folder):
        if html_filename.endswith(".html"):
            html_file_path = os.path.join(combined_html_folder, html_filename)

            # Read HTML content
            with open(html_file_path, "r", encoding="utf-8") as html_file:
                html_content = html_file.read()

            print(f"Converting {html_filename} to JSON...")

            # Convert HTML to JSON
            json_response = html_to_json(html_content)

            # Save JSON response
            json_filename = os.path.splitext(html_filename)[0] + ".json"
            json_file_path = os.path.join(json_output_folder, json_filename)

            if "error" not in json_response:
                with open(json_file_path, "w", encoding="utf-8") as json_file:
                    json.dump(json_response, json_file, indent=4)
                print(f"Saved JSON: {json_file_path}")
            else:
                # Save invalid responses for debugging
                error_json_path = os.path.join(json_output_folder, f"{html_filename}_error.json")
                with open(error_json_path, "w", encoding="utf-8") as error_file:
                    json.dump(json_response, error_file, indent=4)
                print(f"Saved invalid response for debugging: {error_json_path}")

    print("\nAll HTML files have been processed into JSON responses.")


# Define the desired output data structure for key topics
class CombineKeyTopicList(BaseModel):
    combine_key_topics: list = Field(description="list of key topics")

def remove_duplicate_key_topics(key_topics):
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = JsonOutputParser(pydantic_object=CombineKeyTopicList)
    print(f'key_topics before filter: {key_topics}')
    # Define the LLM prompt
    prompt = PromptTemplate(
        template="""Remove the duplicate key topics from the list also which has the same meaning combine them into single topic and return the final list.Not more than 10 topics.only list high level topics.\n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser  # Create the chain

    # Invoke the chain
    try:
        response = chain.invoke({"query": key_topics})
        print(f'key_topics after filter: {key_topics}')
         # Check if response is a dictionary
        if isinstance(response, dict):
            return response.get("combine_key_topics", [])
        else:
            return response.key_topics 
    except Exception as e:
        print(f"Error extracting key topics: {e}")
        return []

# Define the desired output data structure for key topics
class KeyTopicsList(BaseModel):
    key_topics: list = Field(description="list of key topics")

# Function to find key topics from a JSON document
def find_key_topic(json_content):
    """
    Extract key topics from a JSON document using an LLM.
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = JsonOutputParser(pydantic_object=KeyTopicsList)

    # Define the LLM prompt
    prompt = PromptTemplate(
        template="""Find key topics discussed in the document from the document.Response the List of Key topics.\n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser  # Create the chain

    # Invoke the chain
    try:
        response = chain.invoke({"query": json_content})
         # Check if response is a dictionary
        if isinstance(response, dict):
            return response.get("key_topics", [])
        else:
            return response.key_topics 
    except Exception as e:
        print(f"Error extracting key topics: {e}")
        return []

# Function to process all JSON files and combine extracted key topics
def process_json_and_extract_key_topics(json_folder):
    """
    Load all JSON files, extract key topics, and combine them into a single list.
    """
    combined_key_topics = []

    # Traverse all JSON files in the specified folder
    for json_filename in os.listdir(json_folder):
        if json_filename.endswith(".json"):
            json_file_path = os.path.join(json_folder, json_filename)

            # Read JSON content
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                json_content = json_file.read()

            print(f"Extracting key topics from: {json_filename}")

            # Find key topics
            key_topics = find_key_topic(json_content)
            if key_topics:
                combined_key_topics.extend(key_topics)
                print(f"Key topics from {json_filename}: {key_topics}")
            else:
                print(f"No key topics found in {json_filename}")

    print("\nAll key topics have been extracted and combined.")
    
    combined_key_topics = remove_duplicate_key_topics(combined_key_topics)
    # Save combined key topics to a file
    output_file = "combined_key_topics.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"key_topics": combined_key_topics}, f, indent=4)

    print(f"\nCombined key topics saved to {output_file}")
    
    return combined_key_topics

# Paths
JSON_FOLDER = "json_responses"  # Folder containing JSON files
CHROMA_BASE_PATH = "./chroma_embeddings"  # Base path for all embeddings
OUTPUT_CSV = "query_results.csv"  # CSV file for results

# Initialize OpenAI LLM and Embeddings
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# Text splitter for documents
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

def process_json_file(json_file, topics, csv_writer):
    """Load JSON, create embeddings, query topics, and write results to CSV."""
    json_name = os.path.basename(json_file).replace(".json", "")
    chroma_db_path = os.path.join(CHROMA_BASE_PATH, json_name)

    # Create a ChromaDB instance for the specific JSON
    vector_store = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)

    try:
        # Load the JSON file
        loader = JSONLoader(file_path=json_file, jq_schema=".", text_content=False)  # Fix: Ensure proper parsing of dict
        docs = loader.load()
        split_docs = text_splitter.split_documents(docs)
        
        # Add documents to ChromaDB
        vector_store.add_documents(split_docs)
        retriever = vector_store.as_retriever()
        
        # Define tool for the agent
        tools = [
            Tool(
                name="retrieval_tool",
                description="Tool for querying relevant information",
                func=retriever.get_relevant_documents
            )
        ]
        
        # Initialize agent
        agent = initialize_agent(
            tools=tools, 
            llm=llm, 
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True
        )
        
        print(f"Processing: {json_file}")
        
        # Query for each topic
        for topic in topics:
            question = f"What does this document mention about '{topic}'?"
            agent_response = agent.run(question)
            
            # Write results to CSV
            csv_writer.writerow({
                "JSON File": os.path.basename(json_file),
                "Topic": topic,
                "Response": agent_response
            })
            print(f"Queried topic: {topic}")
            time.sleep(0.5)  # Add slight delay to avoid rate limits
    
    finally:
        print(f"Embedding stored for: {json_name}")

def get_the_response_for_key_topics():
    """Main function to process all JSON files and save results to CSV."""
    #pick key topic list from the json file
    # Load key topics from combined_key_topics.json
    with open('combined_key_topics.json', 'r') as f:
        key_topics_data = json.load(f)
        key_topics = key_topics_data['key_topics']
    # Ensure base embedding directory exists
    if not os.path.exists(CHROMA_BASE_PATH):
        os.makedirs(CHROMA_BASE_PATH)

    # Open CSV file for writing
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["JSON File", "Topic", "Response"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        # Process each JSON file
        for json_filename in os.listdir(JSON_FOLDER):
            if json_filename.endswith(".json"):
                json_file_path = os.path.join(JSON_FOLDER, json_filename)
                process_json_file(json_file_path, key_topics, csv_writer)
    
    print(f"All results have been saved to {OUTPUT_CSV}.")


# Function to generate a summary
def generate_summary_response(key_topic, data):
    llm = ChatOpenAI(model='gpt-4o-mini')  # Initialize LLM model
    prompt_template = f"""
    You are tasked with summarizing common and uncommon aspects from the document for the key topics.
    You have been given the response for the key topic: {key_topic}
    and the response from all documents: {data}
    Please provide a concise summary.
    """
    ai_msg = llm.invoke(prompt_template)
    return ai_msg.content


def final_comparison():
    #stcture the response
    # File path
    input_file_path = 'query_results.csv'
    output_file_path = 'restructured_output.csv'
    # Read the file
    # Load the CSV file
    try:
        # Step 1: Load the CSV with no header
        df = pd.read_csv(input_file_path)

        # Step 2: Rename the columns for clarity
        df.columns = ['File', 'Topic', 'Response']

        # Step 3: Pivot the data: Topics as rows, Files as columns, Responses as values
        pivot_df = df.pivot(index='Topic', columns='File', values='Response')

        # Step 4: Reset index for cleaner output
        pivot_df = pivot_df.reset_index()

        # Step 5: Save the restructured DataFrame to a new CSV file
        pivot_df.to_csv(output_file_path, index=False)
        print(f"Restructured data saved to: {output_file_path}")

    except Exception as e:
        print(f"Error: {e}")


    input_file_path_1= 'restructured_output.csv'
    output_file_path_1 = 'document_comparision_output.csv'

    try:
        # Step 1: Load the restructured file
        df = pd.read_csv(input_file_path_1)

        # Step 2: Combine all responses per topic
        def combine_responses(row):
            # Combine all responses (excluding 'Topic' column) into a single string
            return ' '.join(str(response) for response in row[1:] if pd.notna(response))

        # Apply the combination function to get combined responses
        df['Combined Responses'] = df.apply(combine_responses, axis=1)

        # Step 3: Generate the 'Final Summary' for each topic
        df['Final Summary'] = df.apply(
            lambda row: generate_summary_response(row['Topic'], row['Combined Responses']),
            axis=1
        )

        # Step 4: Save the DataFrame with the 'Final Summary' column
        df.to_csv(output_file_path_1, index=False)
        print(f"Final summarized data saved to: {output_file_path_1}")

    except Exception as e:
        print(f"Error: {e}")
