# README

## Overview

This project is designed to analyze the online documentation at docs.oneprocloud.com and upload the results to the Dify knowledge base. The documentation is structured using VuePress.

## Prerequisites

Before running the project, ensure that you have set the necessary environment variables. Currently, we are using Microsoft's OpenAI API for text analysis. 

### Azure OpenAI Environment Variables

- **AZURE_OPENAI_API_KEY**: Required for Azure OpenAI service. This is your API key for authentication.
- **AZURE_OPENAI_DEPLOYMENT**: The name of the Azure OpenAI model deployment.
- **AZURE_OPENAI_ENDPOINT**: The endpoint URL for the Azure OpenAI service.
- **AZURE_OPENAI_API_VERSION**: The version of the Azure OpenAI API.

### Dify Knowledge Base Environment Variables

- **KB_API_KEY**: Required for uploading to the knowledge base. This is your API key for authentication.
  
## Command Line Arguments

The following command line arguments are available:

- `--vuepress-path VUEPRESS_PATH`: This is a required argument that specifies the path to the VuePress documentation directory.
- `--kb-url KB_URL`: This is an optional argument. If provided, the tool will upload the analyzed content to the specified knowledge base URL.
- `--kb-name KB_NAME`: This is also an optional argument. It specifies the name of the knowledge base to which the content will be uploaded.

### Important Notes

- If you do not specify `--kb-url` and `--kb-name`, the tool will only analyze the VuePress project without uploading any content to the knowledge base.
- Ensure that the API key is set in your environment variables to enable uploading to the knowledge base.

## Example Usage: Analyzing the VuePress Documentation without Uploading

```
python kb_builder.py --vuepress-path /path/to/vuepress/docs
```
After running the script, a directory named `kb` will be created at the same level as the `src` directory. This directory will serve as a cache to record documents that have already been uploaded, preventing duplicate calls to the GPT model.

### Cache Management

The `kb` directory will contain two subdirectories:
- `src`: This will store the original markdown files.
- `converted`: This will hold the converted markdown files that have been processed and uploaded to the knowledge base.

The script ensures that if a document has already been uploaded and its content remains unchanged, it will not be uploaded again, thus saving resources and time.

## Example Usage: Uploading to the Knowledge Base

To upload the analyzed content to the knowledge base, you can use the following command:

```
python kb_builder.py --vuepress-path /path/to/vuepress/docs --kb-url https://api.dify.ai/v1/knowledge-bases/your-knowledge-base-id --kb-name "Your Knowledge Base Name"
```

This command will analyze the VuePress documentation, upload the converted markdown files to the specified knowledge base, and create a new knowledge base if it doesn't exist.
