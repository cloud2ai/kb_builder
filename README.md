# README

## Overview

This project provides tools for converting various document formats into text suitable for Dify knowledge base import. It includes support for VuePress documentation and Excel files.

## Prerequisites

Before running the project, ensure that you have set the necessary environment variables. Currently, we are using Microsoft's OpenAI API for text analysis.

### Azure OpenAI Environment Variables

- **AZURE_OPENAI_API_BASE**: Required for Azure OpenAI service. This is your API base URL.
- **AZURE_OPENAI_API_KEY**: Required for Azure OpenAI service. This is your API key for authentication.
- **AZURE_OPENAI_DEPLOYMENT**: Required for Azure OpenAI service. This is your model deployment name (e.g., "gpt-4.1-mini").
- **AZURE_OPENAI_API_VERSION**: Required for Azure OpenAI service. This is your API version (e.g., "2024-10-01-preview").

### Dify Knowledge Base Environment Variables

- **KB_API_KEY**: Required for uploading to the knowledge base. This is your API key for authentication.

## VuePress to Dify Knowledge Base Converter

This tool analyzes VuePress documentation and converts it into text format for importing into Dify knowledge base, supporting multiple languages and automatic content summarization.

### Features

- **Multi-Language Support**: Processes documentation in multiple languages (e.g., English, Chinese)
- **Automatic Content Analysis**: Analyzes markdown files and generates structured content
- **AI-Powered Summarization**: Uses large language models to generate content summaries
- **Knowledge Base Upload**: Direct upload to Dify knowledge base with language-specific organization
- **Cache Management**: Prevents duplicate processing and optimizes performance
- **Debug Mode Support** for detailed logging

### Usage

#### Basic Usage (Analysis Only)

```bash
vuepress2dify --vuepress-path /path/to/vuepress/docs
```

#### Upload to Knowledge Base

```bash
vuepress2dify --vuepress-path /path/to/vuepress/docs --kb-url https://api.dify.ai/v1 --kb-name "docs" --lang "en,zh"
```

#### Specify Languages

```bash
# Process specific languages
vuepress2dify --vuepress-path /path/to/vuepress/docs --lang "en,zh"

# Process all available languages (default)
vuepress2dify --vuepress-path /path/to/vuepress/docs
```

#### Enable Debug Mode

```bash
vuepress2dify --vuepress-path /path/to/vuepress/docs --debug
```

### Command Line Arguments

- `--vuepress-path VUEPRESS_PATH`: Required. Path to the VuePress documentation directory
- `--lang LANG`: Optional. Languages to process (e.g., "en" or "en,zh"). Default: all available languages
- `--kb-url KB_URL`: Optional. Dify knowledge base API URL for uploading content
- `--kb-name KB_NAME`: Optional. Knowledge base name prefix for language-specific bases
- `--debug`: Optional. Enable debug mode with detailed logging
- `--cache-stats`: Optional. Show cache statistics and exit
- `--cache-details`: Optional. Show detailed cache information including original content
- `--clear-cache`: Optional. Clear all cached summaries before processing
- `--prompt`: Optional. Custom prompt for content summarization. This prompt will be used to guide the AI in generating summaries. If not provided, the default prompt will be used. Example: "You are a technical documentation expert. Please generate a concise summary for the following content, focusing on technical points and solutions."

### Processing Details

#### Content Analysis

- Analyzes markdown files in VuePress documentation
- Extracts content structure and metadata
- Generates summaries for each document section
- Maintains hierarchical organization

#### Knowledge Base Organization

- Creates language-specific knowledge bases (e.g., "docs_en", "docs_zh")
- Uploads converted markdown files to appropriate knowledge base
- Cleans up outdated documents automatically
- Maintains content relationships and structure

#### Cache Management

The tool creates a `kb` directory at the same level as the `src` directory containing:
- `.uploaded` files: Records of successfully uploaded documents
- `.metadata.json` files: Document metadata including summaries and structure
- `.txt` files: Cached content for comparison

This ensures that unchanged documents are not re-uploaded, saving resources and time.

### Important Notes

1. VuePress documentation directory must exist and be accessible
2. Markdown files are automatically detected and processed
3. Language detection is based on file structure and naming conventions
4. Knowledge base upload requires valid API credentials
5. Cache is automatically cleaned up to remove outdated files
6. Supports nested documentation structures

### Error Handling

- Displays error message if VuePress path doesn't exist
- Shows detailed error for markdown parsing issues
- API errors don't interrupt content analysis
- Graceful handling of missing or corrupted files
- Continues processing even if individual files fail

## Excel to Dify Knowledge Base Converter

This tool converts Excel files into text format for importing into Dify knowledge base, supporting both horizontal and vertical processing modes with AI-powered content summarization.

### Quick Start

```bash
# Basic usage with default settings
python kb_builder/cmd/excel2dify.py your_file.xlsx

# With custom prompt for better summarization
python kb_builder/cmd/excel2dify.py your_file.xlsx --prompt "You are a technical expert. Please summarize the following content focusing on key solutions."

# Upload to Dify knowledge base
python kb_builder/cmd/excel2dify.py your_file.xlsx --kb-url https://api.dify.ai/v1 --kb-name "excel_data"

# Complete example with all features
python kb_builder/cmd/excel2dify.py data.xlsx --mode horizontal --keep-fields "Problem,Solution" --prompt "You are a support specialist. Please summarize this troubleshooting content." --kb-url https://api.dify.ai/v1 --kb-name "support_data" --debug
```

### Features

- **Horizontal Processing Mode**: First row as headers, each subsequent row becomes a complete paragraph
- **Vertical Processing Mode**: Each column as a section containing all content from that column
- **Multi-Sheet Support**: Each sheet is processed and saved as a separate file with naming format `original_filename-sheet_name.txt`
- **Field Filtering**: Specify fields to keep in output while all fields participate in summarization
- **AI Summarization**: Uses large language models to generate precise summaries for each paragraph, improving retrieval accuracy
- **Custom Prompt Support**: Provide custom prompts for better context-aware summarization
- **Automatic Text File Generation**
- **Debug Mode Support** for detailed logging

### Installation Dependencies

```bash
pip install pandas openpyxl
```

### Usage

#### Basic Usage

```bash
python kb_builder/cmd/excel2dify.py path/to/your/excel_file.xlsx
```

#### Specify Processing Mode

```bash
# Horizontal processing mode (default)
python kb_builder/cmd/excel2dify.py path/to/your/excel_file.xlsx --mode horizontal

# Vertical processing mode
python kb_builder/cmd/excel2dify.py path/to/your/excel_file.xlsx --mode vertical
```

#### Specify Fields to Keep

```bash
# Keep only specified fields in output, but all fields participate in summarization
python kb_builder/cmd/excel2dify.py path/to/your/excel_file.xlsx --keep-fields "Problem,Solution"
```

#### Specify Output File

```bash
python kb_builder/cmd/excel2dify.py path/to/your/excel_file.xlsx --output custom_output.txt
```

#### Enable Debug Mode

```bash
python kb_builder/cmd/excel2dify.py path/to/your/excel_file.xlsx --debug
```

#### Custom Prompt for Summarization

```bash
# Use custom prompt for better context-aware summarization
python kb_builder/cmd/excel2dify.py path/to/your/excel_file.xlsx --prompt "You are a technical documentation expert. Please generate a concise summary for the following content, focusing on technical points and solutions."
```

#### Complete Example

```bash
# Horizontal mode, keep problem and solution fields, custom prompt, enable debug
python kb_builder/cmd/excel2dify.py printer.xlsx --mode horizontal --keep-fields "Problem,Solution" --prompt "You are a technical support expert. Please summarize the following troubleshooting content." --debug
```

### Processing Mode Details

#### Horizontal Processing Mode

- First row serves as header row
- Each subsequent row becomes a complete paragraph
- Format: Each header: content on separate lines, single newlines within paragraphs
- Double newlines separate paragraphs
- AI-generated summary added at the end of each paragraph

**Example Excel:**
```
Name    Age     Profession
John    25      Engineer
Jane    30      Designer
```

**Output Text:**
```
## Name
John
Jane

## Age
25
30

## Profession
Engineer
Designer
```

#### Vertical Processing Mode

- Each column represents a section
- Column headers become section titles
- All non-empty content from that column becomes section content

**Example Excel:**
```
Name    Age     Profession
John    25      Engineer
Jane    30      Designer
```

**Output Text:**
```
## Name
John
Jane

## Age
25
30

## Profession
Engineer
Designer
```

### Field Filtering Functionality

Use the `--keep-fields` parameter to specify which fields to retain in the output:

- All fields participate in AI summarization generation
- Only specified fields appear in the final text output
- Field names separated by commas, e.g., `"问题,处理方式,解决方案"`

This feature is particularly useful for:
- Reducing output file size
- Focusing on the most important information
- Maintaining summary completeness (based on all fields)

### AI Summarization Feature

Each paragraph automatically generates a summary:

- Uses Azure OpenAI GPT-4 model
- Summary limited to 100 characters
- Includes key terms and important information
- Format: `[总结] summary content`
- Follows paragraph content immediately for easy retrieval
- Supports custom prompts for better context-aware summarization
- Default prompt focuses on technical content analysis
- Custom prompts can be tailored for specific domains or use cases

### Cache Mechanism

The tool implements an intelligent caching system to avoid redundant LLM API calls and improve processing efficiency:

#### How It Works

1. **Content Hashing**: Generates SHA256 hash for each content piece that needs summarization
2. **Cache Key**: Uses `sheet_name_content_hash` as the cache key
3. **Cache Loading**: Loads existing cache at start for potential hits during processing
4. **Cache Overwrite**: Clears cache after processing to implement overwrite behavior for next run
5. **Content Preservation**: Includes both original content and processed text for debugging

#### Cache File

- **Location**: `{excel_filename}_metadata.json` in the same directory as the Excel file
- **Format**: JSON format containing summary results, content hashes, metadata, original content, and processed text for debugging

#### Cache Structure Example

For input file `printer_manual.xlsx`, cache file `printer_manual_metadata.json`:

```json
{
  "Sheet1_a1b2c3d4e5f6...": {
    "summary": "Printer troubleshooting guide",
    "content_hash": "a1b2c3d4e5f6...",
    "sheet_name": "Sheet1",
    "original_content": "问题：打印机无法打印\n处理方式：检查连接线\n状态：已解决",
    "processed_text": "问题：打印机无法打印\n处理方式：检查连接线\n[总结] Printer troubleshooting guide"
  }
}
```

#### Cache File Naming Examples

- Input file: `printer_manual.xlsx` → Cache file: `printer_manual_metadata.json`
- Input file: `data.xlsx` → Cache file: `data_metadata.json`
- Input file: `report_2024.xlsx` → Cache file: `report_2024_metadata.json`

#### Cache Management Commands

```bash
# Show cache statistics
python kb_builder/cmd/excel2dify.py excel_file.xlsx --cache-stats

# Show detailed cache information including original content
python kb_builder/cmd/excel2dify.py excel_file.xlsx --cache-details

# Clear all cache
python kb_builder/cmd/excel2dify.py excel_file.xlsx --clear-cache

# Debug mode (shows cache information)
python kb_builder/cmd/excel2dify.py excel_file.xlsx --debug
```

#### Use Cases

1. **Cache Hits During Processing**: Uses existing cache entries for unchanged content during the same processing run
2. **Fresh Processing**: Each new run starts with clean cache, ensuring consistency
3. **Content Tracking**: Both original Excel content and final processed text are preserved
4. **Debug Support**: Complete processing history available for troubleshooting
5. **Processing Analysis**: Analyze how Excel data is transformed into final text

#### Performance Benefits

- **Cache Hits**: Uses existing cache entries during processing for unchanged content
- **Fresh Start**: Each new run starts with clean cache, ensuring consistency
- **Content Preservation**: Both original and processed content available for analysis
- **Storage Optimization**: Each cache entry ~500-1200 bytes (includes both original and processed content)
- **Debug Support**: Complete processing history preserved for troubleshooting and verification

### Multi-Sheet Processing

- Automatically detects all sheets in Excel file
- Each sheet processed and saved as independent file
- File naming format: `original_filename-sheet_name.txt`
- Supports Chinese sheet names (automatically converted to safe filenames)

### Important Notes

1. Excel file must exist and be readable
2. Empty cells are automatically skipped
3. Output files saved in same directory as Excel file by default
4. Existing output files will be overwritten
5. Supports Chinese content processing
6. AI summarization requires Azure OpenAI API configuration
7. If LLM service unavailable, summarization is skipped but content processing continues
8. Cache file location: Same directory as Excel file, named `{excel_filename}_metadata.json`
9. Cache behavior: Loads existing cache for hits during processing, clears after completion
10. Content preservation: Both original Excel content and processed text are saved

### Error Handling

- Displays error message if file doesn't exist
- Shows detailed error for Excel file format issues
- Empty files generate empty txt files
- LLM service errors don't interrupt overall processing
- Processing errors for individual sheets don't affect other sheets

### Performance Optimization

- Supports caching mechanism to avoid duplicate processing
- Intelligently skips empty sheets
- Batch processes multiple sheets
- Optional debug mode for troubleshooting

### Command Line Arguments

- `excel_path`: Required. Path to the Excel file to process
- `--mode`: Optional. Processing mode ('horizontal' or 'vertical'). Default: horizontal
- `--keep-fields`: Optional. Comma-separated list of field names to keep in output
- `--output`: Optional. Output text file path (auto-generated if not provided)
- `--kb-url`: Optional. Knowledge base API endpoint URL
- `--kb-name`: Optional. Name of the knowledge base to create or update
- `--debug`: Optional. Enable debug mode with detailed logging
- `--cache-stats`: Optional. Show cache statistics and exit
- `--cache-details`: Optional. Show detailed cache information including original content
- `--clear-cache`: Optional. Clear all cached summaries before processing
- `--prompt`: Optional. Custom prompt for content summarization. This prompt will be used to guide the AI in generating summaries. If not provided, the default prompt will be used. Example: "You are a technical documentation expert. Please generate a concise summary for the following content, focusing on technical points and solutions."

## Usage Guidelines and Restrictions

### Permitted Uses
1. Personal and commercial use of the tool
2. Modification and distribution of the tool
3. Integration with other systems
4. Creation of derivative works

### Restrictions
1. **Attribution Requirement**: You must include the original copyright notice and license in any copy or substantial portion of the software.
2. **API Usage**: When using this tool, you must comply with the terms of service of the underlying APIs (Azure OpenAI and Dify).
3. **Commercial Use**: While commercial use is permitted, you must:
   - Not claim ownership of the original software
   - Not use the OneProCloud name or branding without explicit permission
   - Not redistribute the software under a different license
4. **Modifications**: If you modify the software, you must:
   - Clearly indicate the changes made
   - Include the original copyright notice
   - Document any significant changes

### Best Practices
1. Keep your API keys secure and never commit them to version control
2. Regularly update the tool to get the latest features and security fixes
3. Report any bugs or issues through the issue tracker
4. Consider contributing improvements back to the project

## Contributing

We welcome contributions to this project! Please read the following guidelines before contributing.

### Contributor License Agreement (CLA)

By contributing to this project, you agree to the following terms:

1. **Copyright License**: You grant OneProCloud a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare derivative works of, publicly display, publicly perform, sublicense, and distribute your contributions.

2. **Patent License**: You grant OneProCloud a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer your contributions.

3. **Representations**: You represent that:
   - You are legally entitled to grant the above licenses
   - Your contributions are your original work
   - Your contributions do not violate any third party's rights
   - Your contributions do not contain any malicious code

### How to Contribute

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Write or update tests as needed
5. Update documentation
6. Submit a pull request

### Code Style and Quality

1. Follow PEP 8 style guide for Python code
2. Write clear, descriptive commit messages
3. Include tests for new features
4. Update documentation for any changes
5. Ensure all tests pass before submitting

### Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation with any new features or changes
3. The PR will be merged once it has been reviewed and approved

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
