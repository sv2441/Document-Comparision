import streamlit as st
import os
import shutil
import json
from pathlib import Path
import pandas as pd

# Import backend processing functions
from app import (
    process_pdfs_in_folder,
    process_images_to_htmls,
    combine_html_files,
    process_combined_html_to_json,
    process_json_and_extract_key_topics,
    get_the_response_for_key_topics,
    final_comparison
)

# Write the Google Cloud credentials to a temporary file
with open("google_credentials.json", "w") as f:
    f.write(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_2"])

# Set the environment variable for Google credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_credentials.json"
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

# Define constants
UPLOAD_FOLDER = "UCB PP"
OUTPUT_FILE = "document_comparision_output.csv"

def reset_environment():
    """Reset all generated data, embeddings, and directories."""
    # List of folders and files to clear
    folders_to_clear = ["UCB PP", "Images", "HTMLs", "Combined_HTMLs", "json_responses", "chroma_embeddings"]
    files_to_clear = [
        "query_results.csv",
        "restructured_output.csv",
        "document_comparision_output.csv",
        "combined_key_topics.json",
    ]

    # Clear folders
    for folder in folders_to_clear:
        shutil.rmtree(folder, ignore_errors=True)

    # Clear individual files
    for file in files_to_clear:
        if Path(file).exists():
            os.remove(file)

def display_download_and_reset(file_path, label):
    """Display a download button and reset environment after download."""
    if Path(file_path).exists():
        # Use `open` and immediately close the file after reading
        with open(file_path, "rb") as file:
            file_data = file.read()

        # Show the download button
        download_clicked = st.download_button(
            label=label,
            data=file_data,
            file_name=Path(file_path).name,
            mime="text/csv",
        )

        # Reset environment after the download button is clicked
        if download_clicked:
            reset_environment()
            st.success("Environment reset successfully!")


def main():
    st.title("Automated PDF Processing and Analysis Tool")
    st.markdown("Upload PDFs, process them, and download a comprehensive comparison.")

    # File Upload Section
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader("Upload multiple PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        for file in uploaded_files:
            file_path = os.path.join(UPLOAD_FOLDER, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
        st.success("Files uploaded successfully!")

    # Processing Section
    st.header("Start Processing")
    if st.button("Run All Steps"):
        progress = st.progress(0)
        status = st.empty()

        try:
            # Step 1: Process PDFs to Images
            status.text("Processing PDFs into images...")
            process_pdfs_in_folder()
            progress.progress(20)

            # Step 2: Convert Images to HTML
            status.text("Converting images into HTML...")
            process_images_to_htmls()
            progress.progress(40)

            # Step 3: Combine HTML Files
            status.text("Combining HTML files...")
            combine_html_files()
            progress.progress(60)

            # Step 4: Convert Combined HTML to JSON
            status.text("Converting HTML to JSON...")
            process_combined_html_to_json()
            progress.progress(80)

            # Step 5: Extract Key Topics and Generate Outputs
            status.text("Extracting key topics and generating final outputs...")
            process_json_and_extract_key_topics("json_responses")
            get_the_response_for_key_topics()
            final_comparison()
            progress.progress(100)

            st.success("All steps completed successfully!")
            
            # Display the final DataFrame
            if Path(OUTPUT_FILE).exists():
                st.header("Final Comparison Data")
                df = pd.read_csv(OUTPUT_FILE)
                st.dataframe(df)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Download and Reset Section
    st.header("Download Results")
    display_download_and_reset(OUTPUT_FILE, "Download Final Comparison CSV")

if __name__ == "__main__":
    main()
