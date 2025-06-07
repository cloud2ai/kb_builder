# README

## Overview

This project is designed to analyze the online documentation at docs.oneprocloud.com and upload the results to the Dify knowledge base. The documentation is structured using VuePress.

## Prerequisites

Before running the project, ensure that you have set the necessary environment variables. Currently, we are using Microsoft's OpenAI API for text analysis.

### Azure OpenAI Environment Variables

- **AZURE_OPENAI_API_BASE**: Required for Azure OpenAI service. This is your API base URL.
- **AZURE_OPENAI_API_KEY**: Required for Azure OpenAI service. This is your API key for authentication.
- **AZURE_OPENAI_DEPLOYMENT**: Required for Azure OpenAI service. This is your model deployment name (e.g., "gpt-4.1-mini").
- **AZURE_OPENAI_API_VERSION**: Required for Azure OpenAI service. This is your API version (e.g., "2024-10-01-preview").

### Dify Knowledge Base Environment Variables

- **KB_API_KEY**: Required for uploading to the knowledge base. This is your API key for authentication.

## Command Line Arguments

The following command line arguments are available:

- `--vuepress-path VUEPRESS_PATH`: This is a required argument that specifies the path to the VuePress documentation directory.
- `--lang LANG`: Optional argument to specify which languages to process (e.g., "en" or "en,zh"). If not specified, all available languages will be processed.
- `--kb-url KB_URL`: Optional argument. If provided, the tool will upload the analyzed content to the specified knowledge base URL.
- `--kb-name KB_NAME`: Optional argument. It specifies the name of the knowledge base to which the content will be uploaded. This will be used as the prefix for language-specific knowledge bases (e.g., "docs_en" and "docs_zh").
- `--debug`: Optional flag to enable debug mode with detailed logging.

### Important Notes

- If you do not specify `--kb-url` and `--kb-name`, the tool will only analyze the VuePress project without uploading any content to the knowledge base.
- Ensure that the API key is set in your environment variables to enable uploading to the knowledge base.
- The tool supports multiple languages and will create language-specific knowledge bases when uploading.

## Example Usage: Analyzing the VuePress Documentation without Uploading

```
vuepress2dify --vuepress-path /path/to/vuepress/docs
```

After running the script, a directory named `kb` will be created at the same level as the `src` directory. This directory will serve as a cache to record documents that have already been uploaded, preventing duplicate calls to the GPT model.

### Cache Management

The `kb` directory will contain language-specific subdirectories (e.g., `en`, `zh`). Each language directory contains:
- `.uploaded` files: Records of successfully uploaded documents
- `.metadata.json` files: Document metadata including summaries and structure
- `.txt` files: Cached content for comparison

The script ensures that if a document has already been uploaded and its content remains unchanged, it will not be uploaded again, thus saving resources and time. The cache is automatically cleaned up to remove files that are no longer part of the documentation.

## Example Usage: Uploading to the Knowledge Base

To upload the analyzed content to the knowledge base, you can use the following command:

```
vuepress2dify --vuepress-path /path/to/vuepress/docs --kb-url https://api.dify.ai/v1 --kb-name "docs" --lang "en,zh"
```

This command will:
1. Analyze the VuePress documentation for English and Chinese content
2. Create language-specific knowledge bases (e.g., "docs_en" and "docs_zh")
3. Upload the converted markdown files to the appropriate knowledge base
4. Clean up any outdated documents from the knowledge base
5. Maintain a local cache to optimize future updates
