#!/usr/bin/env python3
"""
Command line interface for Excel to Dify Knowledge Base Converter.

This tool helps convert Excel files into text format for Dify knowledge
base. Supports both horizontal and vertical processing modes with optional
field filtering and AI-powered content summarization.

Cache Mechanism:
- Generates {excel_filename}.metadata.json file in the same directory as the Excel file
- Caches LLM-generated summaries to avoid redundant API calls during processing
- Uses content hash as cache key to detect changes
- Loads existing cache at start for potential hits during processing
- Clears cache after processing to implement overwrite behavior for next run
- Includes original content and processed text for debugging and troubleshooting
- Cache structure:
  {
    "sheet_name_content_hash": {
      "summary": "Generated summary text",
      "content_hash": "sha256_hash_of_content",
      "sheet_name": "Sheet1",
      "original_content": "original_content_text",
      "processed_text": "final_processed_text_with_summary"
    }
  }
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from kb_builder.utils.excel_processor import ExcelProcessor
from kb_builder.utils.kb_client import KBClient

# Initialize logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description=(
            'Excel to Dify Knowledge Base Converter\n'
            'This tool converts Excel files into text format for Dify knowledge base.\n'
            'Supports both horizontal and vertical processing modes with optional field\n'
            'filtering and AI-powered content summarization.'
        )
    )
    parser.add_argument(
        'excel_path',
        help='Path to the Excel file to process'
    )
    parser.add_argument(
        '--mode',
        choices=['horizontal', 'vertical', 'sheet-horizontal', 'sheet-vertical'],
        default='horizontal',
        help=(
            'Processing mode: horizontal (first row as headers, each row as '
            'paragraph), vertical (each column as section), sheet-horizontal '
            '(row-wise description with background context), or sheet-vertical '
            '(column-wise description with background context). Default: horizontal'
        )
    )
    parser.add_argument(
        '--keep-fields',
        help=(
            'Comma-separated list of field names to keep in output. '
            'All fields will be used for summarization, but only specified '
            'fields will appear in the final text. Example: "问题,处理方式"'
        )
    )
    parser.add_argument(
        '--output',
        help=(
            'Output text file path (optional, will generate automatically if '
            'not provided)'
        )
    )
    parser.add_argument(
        '--kb-url',
        help=(
            'Knowledge base API endpoint URL. '
            'Example: https://api.dify.ai/v1. '
            'If not provided, the tool will only process files without '
            'uploading to knowledge base.'
        )
    )
    parser.add_argument(
        '--kb-name',
        help=(
            'Name of the knowledge base to create or update. '
            'This will be used as the prefix for Excel-specific knowledge '
            'bases. For example, if set to "excel_data", it will create '
            '"excel_data" knowledge base for the Excel content.'
        )
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with detailed logging'
    )
    parser.add_argument(
        '--cache-stats',
        action='store_true',
        help='Show cache statistics and exit'
    )
    parser.add_argument(
        '--cache-details',
        action='store_true',
        help='Show detailed cache information including original content'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cached summaries before processing'
    )
    parser.add_argument(
        '--prompt',
        help=(
            'Custom prompt for content summarization. This prompt will be '
            'used to guide the AI in generating summaries. If not provided, '
            'the default prompt will be used. Example: "You are a technical '
            'documentation expert. Please generate a concise summary for the '
            'following content, focusing on technical points and solutions."'
        )
    )
    parser.add_argument(
        '--background-info',
        help=(
            'Background information to provide context for analysis. '
            'This is particularly useful for sheet-horizontal and sheet-vertical '
            'modes to provide domain-specific context. Example: "This data '
            'represents customer support tickets with technical issues and '
            'their resolution status."'
        )
    )

    return parser.parse_args()


def setup_logging(debug: bool = False):
    """
    Configure logging based on debug mode.

    Args:
        debug: Whether to enable debug mode
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    if debug:
        logger.debug("Debug mode enabled")

    return logger


def main():
    """
    Main entry point for the Excel to Dify converter.

    Returns:
        0 on success, 1 on error
    """
    args = parse_arguments()
    logger = setup_logging(args.debug)

    try:
        # Initialize Excel processor
        processor = ExcelProcessor(args.excel_path, args.keep_fields, args.prompt, args.background_info)

        # Handle cache statistics request
        if args.cache_stats:
            stats = processor.get_cache_stats()
            print("Cache Statistics:")
            print(f"  Total entries: {stats.get('total_entries', 0)}")
            print(f"  Unique sheets: {stats.get('unique_sheets', 0)}")
            print(f"  Total summaries: {stats.get('total_summaries', 0)}")
            print(f"  Sheet-horizontal entries: {stats.get('sheet_horizontal_entries', 0)}")
            print(f"  Sheet-vertical entries: {stats.get('sheet_vertical_entries', 0)}")
            print(f"  Traditional entries: {stats.get('traditional_entries', 0)}")
            print(f"  Cache file: {stats.get('cache_file', 'N/A')}")
            print(f"  Cache type: {stats.get('cache_type', 'N/A')}")
            return 0

        # Handle cache details request
        if args.cache_details:
            details = processor.get_cache_details()
            print("Cache Details:")
            print(f"  Cache file: {details.get('cache_file', 'N/A')}")
            print(f"  Total entries: {details.get('total_entries', 0)}")
            print("\nEntries:")
            for key, entry in details.get('entries', {}).items():
                print(f"  Key: {key}")
                print(f"    Sheet: {entry.get('sheet_name', 'N/A')}")
                print(f"    Content Hash: {entry.get('content_hash', 'N/A')}")
                print(f"    Summary: {entry.get('summary', 'N/A')}")
                print(f"    Original Content: {entry.get('original_content', 'N/A')[:100]}...")
                print(f"    Processed Text: {entry.get('processed_text', 'N/A')[:100]}...")
                print(f"    Content Length: {entry.get('content_length', 0)}")
                print(f"    Processed Length: {entry.get('processed_length', 0)}")
                print()
            return 0

        # Handle cache clear request
        if args.clear_cache:
            processor.clear_cache()
            print("Cache cleared successfully")
            return 0

        # Process all sheets
        results = processor.process_all_sheets(args.mode)

        # Save each sheet to separate text file
        saved_files = []
        failed_sheets = []
        for sheet_name, content, output_path in results:
            if content.strip():  # Only save non-empty content
                saved_path = processor.save_sheet_to_txt(content, output_path)
                saved_files.append((sheet_name, saved_path))
                logger.info(f"Saved sheet '{sheet_name}' to: {saved_path}")
            else:
                logger.warning(f"Skipping empty sheet: {sheet_name}")
                if args.background_info:  # This is a sheet analysis mode
                    failed_sheets.append(sheet_name)

        # Report results
        if saved_files:
            logger.info(
                f"Successfully converted {len(saved_files)} sheets to text files"
            )

        if failed_sheets:
            logger.error(
                f"LLM analysis failed for {len(failed_sheets)} sheets: "
                f"{', '.join(failed_sheets)}"
            )
            print(
                f"Warning: LLM analysis failed for {len(failed_sheets)} sheets. "
                f"No output files generated for these sheets."
            )

        # Clear cache after processing to implement overwrite behavior
        processor._clear_cache_after_processing()

        # Upload to knowledge base if specified
        if args.kb_url and args.kb_name:
            logger.info("Starting upload to knowledge base...")

            # Initialize KB client
            kb_client = KBClient(
                kb_name=args.kb_name,
                base_url=args.kb_url
            )
            kb_client.create_kb_metadata()

            # Upload each saved file
            uploaded_count = 0
            for sheet_name, file_path in saved_files:
                if processor.upload_to_kb(sheet_name, file_path, kb_client):
                    uploaded_count += 1

            logger.info(
                f"Successfully uploaded {uploaded_count} sheets to knowledge base"
            )
            print(
                f"Upload completed! Uploaded {uploaded_count} sheets to "
                f"knowledge base: {args.kb_name}"
            )

        print(f"Conversion completed! Generated {len(saved_files)} files:")
        for sheet_name, file_path in saved_files:
            print(f"  - {file_path}")

        if failed_sheets:
            print(f"\nNote: {len(failed_sheets)} sheets failed LLM analysis and were skipped.")

    except Exception as e:
        logger.exception(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())