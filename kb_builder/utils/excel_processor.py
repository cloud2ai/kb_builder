#!/usr/bin/env python3
"""
Excel processing utilities for converting Excel files to text format.

This module provides utilities for processing Excel files with support for:
- Horizontal and vertical processing modes
- Field filtering
- AI-powered content summarization with caching
- Multi-sheet processing
"""

import hashlib
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from devtoolbox.llm.azure_openai_provider import AzureOpenAIConfig
from devtoolbox.llm.service import LLMService
from kb_builder.utils.kb_client import KBClient

# Initialize logging
logger = logging.getLogger(__name__)

# Summary prompt for Excel content
EXCEL_SUMMARY_PROMPT = """
You are a professional technical content analyzer.

Your task is to generate a concise summary (max 300 characters) for a piece of
technical content extracted from an Excel file. The summary should:

- Capture the main topic or issue being addressed
- Include key technical terms or keywords
- Be clear and searchable for RAG systems
- Focus on the most important information
- Use relevant terminology from the content

Return only the summary text, without any additional formatting or explanation.
"""

SHEET_ANALYSIS_PROMPT = """
Please analyze the Excel sheet according to these rules:

- If background information is provided, use it as the main context to
  understand the sheet's structure.
- Otherwise, determine the structure by checking:
  - If the first row has multiple distinct column headers and the other rows
    have similar data, treat the sheet as horizontally organized (each row is
    a data record).
  - If the first column has multiple field names and the other columns have
    related values, treat the sheet as vertically organized (each column is a
    data entity).

Based on the structure:
- For horizontal structure: analyze the sheet row by row. For each row,
  rewrite the data as a natural-language sentence that fully expresses its
  meaning, using the headers and any background information if available.
- For vertical structure: analyze the sheet column by column. For each
  column, rewrite the data as a natural-language sentence that fully
  expresses its meaning, using the field names and any background
  information if available.

The rewritten sentences must:
- Preserve all meaningful information from the original data, without
  omitting anything.
- Use natural, fluent language suitable for downstream RAG (Retrieval-
  Augmented Generation) tasks.
- Be written in the original language detected in the Excel content (do not
  translate or change the language).
- Appear as one sentence per row or column, with one blank line between
  each sentence.
- Be returned as plain text only, with no markdown, no bullet points, and
  no code formatting.
- Do not include any explanations or extra text.

Background information: {background_info}
"""


class ExcelProcessor:
    """
    Process Excel files and convert them to text format for Dify knowledge base.

    This class handles:
    - Reading Excel files with multiple sheets
    - Processing content in horizontal or vertical modes
    - Field filtering for output
    - AI-powered content summarization with caching
    - Saving results to text files
    """

    def __init__(self, excel_path: str, keep_fields: Optional[str] = None,
                 custom_prompt: Optional[str] = None, background_info: Optional[str] = None):
        """
        Initialize the Excel processor.

        Args:
            excel_path: Path to the Excel file to process
            keep_fields: Comma-separated list of field names to keep in output.
                        All fields will be used for summarization, but only
                        specified fields will appear in the final text.
            custom_prompt: Custom prompt for content summarization. If not provided,
                          the default prompt will be used.
            background_info: Background information to provide context for analysis.
                           Used in sheet-horizontal and sheet-vertical modes.
        """
        self.excel_path = Path(excel_path)
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")

        # Parse keep fields
        self.keep_fields = None
        if keep_fields:
            self.keep_fields = [
                field.strip() for field in keep_fields.split(',')
            ]
            logger.info(f"Keep fields: {self.keep_fields}")

        # Set custom prompt
        self.custom_prompt = custom_prompt
        if custom_prompt:
            logger.info("Using custom prompt for summarization")
        else:
            logger.info("Using default prompt for summarization")

        # Set background information
        self.background_info = background_info
        if background_info:
            logger.info("Background information provided for analysis")

        logger.info(f"Initialized Excel processor for: {excel_path}")

        # Initialize LLM service for summarization
        self.llm_service = self._init_llm()

        # Initialize cache - load existing cache for potential hits during processing
        # For sheet-horizontal and sheet-vertical modes, use separate cache files
        if background_info:
            # Create a hash of background info for cache file naming
            bg_hash = hashlib.sha256(
                background_info.encode('utf-8')
            ).hexdigest()[:8]
            cache_filename = (
                f"{self.excel_path.stem}_sheet_analysis_{bg_hash}.metadata.json"
            )
        else:
            cache_filename = f"{self.excel_path.stem}.metadata.json"

        self.cache_file = self.excel_path.parent / cache_filename
        # Load existing cache for potential hits
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """
        Load cache from metadata.json file.

        Returns:
            Cache dictionary or empty dict if file doesn't exist
        """
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                logger.info(f"Loaded cache from: {self.cache_file}")
                return cache
            else:
                logger.info("No cache file found, starting fresh")
                return {}
        except Exception as e:
            logger.warning(f"Error loading cache: {str(e)}")
            return {}

    def _save_cache(self):
        """
        Save cache to metadata.json file.
        """
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            logger.debug(f"Cache saved to: {self.cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")

    def _get_content_hash(self, content: str) -> str:
        """
        Generate hash for content to use as cache key.

        Args:
            content: Content to hash

        Returns:
            SHA256 hash of the content
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _get_cache_key(self, sheet_name: str, content_hash: str) -> str:
        """
        Generate cache key for a specific sheet and content.

        Args:
            sheet_name: Name of the sheet
            content_hash: Hash of the content

        Returns:
            Cache key string
        """
        return f"{sheet_name}_{content_hash}"

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            total_entries = len(self.cache)
            unique_sheets = set()
            total_summaries = 0
            sheet_horizontal_count = 0
            sheet_vertical_count = 0
            traditional_count = 0

            for key, value in self.cache.items():
                sheet_name = value.get('sheet_name', '')
                if sheet_name:
                    unique_sheets.add(sheet_name)

                # Count different types of entries
                mode = value.get('mode', '')
                if mode == 'sheet_horizontal':
                    sheet_horizontal_count += 1
                elif mode == 'sheet_vertical':
                    sheet_vertical_count += 1
                else:
                    traditional_count += 1

                if value.get('summary'):
                    total_summaries += 1

            # Determine cache type based on filename
            cache_type = (
                "sheet_analysis" if "sheet_analysis" in str(self.cache_file)
                else "traditional"
            )

            return {
                'total_entries': total_entries,
                'unique_sheets': len(unique_sheets),
                'total_summaries': total_summaries,
                'sheet_horizontal_entries': sheet_horizontal_count,
                'sheet_vertical_entries': sheet_vertical_count,
                'traditional_entries': traditional_count,
                'cache_file': str(self.cache_file),
                'cache_type': cache_type
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}

    def get_cache_details(self, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed cache information for debugging.

        Args:
            sheet_name: Optional sheet name to filter results

        Returns:
            Dictionary with detailed cache information
        """
        try:
            details = {
                'cache_file': str(self.cache_file),
                'total_entries': len(self.cache),
                'entries': {}
            }

            for key, value in self.cache.items():
                entry_sheet = value.get('sheet_name', '')

                # Filter by sheet name if specified
                if sheet_name and entry_sheet != sheet_name:
                    continue

                details['entries'][key] = {
                    'sheet_name': entry_sheet,
                    'content_hash': value.get('content_hash', ''),
                    'summary': value.get('summary', ''),
                    'original_content': value.get('original_content', ''),
                    'processed_text': value.get('processed_text', ''),
                    'content_length': len(value.get('original_content', '')),
                    'processed_length': len(value.get('processed_text', ''))
                }

            return details
        except Exception as e:
            logger.error(f"Error getting cache details: {str(e)}")
            return {}

    def clear_cache(self):
        """
        Clear all cache entries.
        """
        try:
            self.cache.clear()
            if self.cache_file.exists():
                self.cache_file.unlink()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

    def _clear_cache_after_processing(self):
        """
        Clear cache after processing is complete to implement overwrite behavior.
        """
        try:
            # Save current cache state
            self._save_cache()
            # Then clear it for next run
            self.cache.clear()
            logger.debug("Cache cleared after processing for overwrite behavior")
        except Exception as e:
            logger.error(
                f"Error clearing cache after processing: {str(e)}"
            )

    def _init_llm(self) -> Optional[LLMService]:
        """
        Initialize LLM service for content summarization.

        Returns:
            LLMService instance or None if initialization fails
        """
        try:
            # Use larger max_tokens for sheet analysis modes
            # GPT-4o-mini supports up to 80000 tokens
            max_tokens = 10000
            openai_config = AzureOpenAIConfig(
                temperature=0.1,
                max_tokens=max_tokens
            )
            logger.debug(f"LLM service initialized with max_tokens={max_tokens}")
            return LLMService(openai_config)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM service: {str(e)}")
            return None

    def _summarize_content(self, content: str, sheet_name: str) -> str:
        """
        Generate a summary for the given content using LLM with caching.

        Args:
            content: The content to summarize
            sheet_name: Name of the sheet for cache key

        Returns:
            Summary text or empty string if LLM service is not available
        """
        if not self.llm_service:
            logger.warning(
                "LLM service not available, skipping summarization"
            )
            return ""

        # Generate content hash and cache key
        content_hash = self._get_content_hash(content)
        cache_key = self._get_cache_key(sheet_name, content_hash)

        # Check if summary exists in cache
        if cache_key in self.cache:
            cached_summary = self.cache[cache_key].get('summary', '')
            if cached_summary:
                logger.debug(f"Using cached summary for {sheet_name}")
                return cached_summary

        try:
            messages = [
                {
                    "role": "system",
                    "content": self.custom_prompt or EXCEL_SUMMARY_PROMPT
                },
                {
                    "role": "user",
                    "content": content
                }
            ]

            summary = self.llm_service.chat(messages)
            summary = summary.strip()

            # Cache the result
            self.cache[cache_key] = {
                'summary': summary,
                'content_hash': content_hash,
                'sheet_name': sheet_name,
                'original_content': content,
                'processed_text': summary
            }
            self._save_cache()

            logger.debug(f"Generated and cached summary: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return ""

    def get_sheet_names(self) -> List[str]:
        """
        Get all sheet names from the Excel file.

        Returns:
            List of sheet names
        """
        try:
            excel_file = pd.ExcelFile(self.excel_path)
            sheet_names = excel_file.sheet_names
            logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")
            return sheet_names
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            raise

    def process_sheet_horizontal(self, sheet_name: str) -> str:
        """
        Process a single sheet in horizontal mode.

        In horizontal mode:
        - First row is treated as headers
        - Each subsequent row becomes a complete paragraph
        - All fields are used for summarization
        - Only keep_fields (if specified) appear in output
        - Each paragraph gets a summary at the end

        Args:
            sheet_name: Name of the sheet to process

        Returns:
            Formatted text content
        """
        logger.info(f"Processing sheet '{sheet_name}' in horizontal mode")

        try:
            # Read specific sheet
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            logger.debug(
                f"Sheet '{sheet_name}' loaded with {len(df)} rows and "
                f"{len(df.columns)} columns"
            )

            if df.empty:
                logger.warning(f"Sheet '{sheet_name}' is empty")
                return ""

            # Get headers from first row
            headers = df.columns.tolist()
            logger.debug(f"Headers for sheet '{sheet_name}': {headers}")

            # Filter headers for output if keep_fields is specified
            output_headers = headers
            if self.keep_fields:
                output_headers = [
                    h for h in headers if h in self.keep_fields
                ]
                logger.info(f"Output headers (filtered): {output_headers}")

            paragraphs = []

            # Process each row (skip header row)
            for index, row in df.iterrows():
                # Create full content for summarization (all fields)
                full_content_parts = []
                for header, value in zip(headers, row):
                    if pd.notna(value):  # Skip empty values
                        value_str = str(value).strip()
                        if value_str:
                            full_content_parts.append(
                                f"{header}：{value_str}"
                            )

                # Create output content (only keep fields)
                output_parts = []
                for header, value in zip(headers, row):
                    # Skip empty values and non-keep fields
                    if (pd.notna(value) and header in output_headers):
                        value_str = str(value).strip()
                        if value_str:
                            output_parts.append(f"{header}：{value_str}")

                # Generate summary using full content
                if full_content_parts:
                    full_content = "\n".join(full_content_parts)
                    summary = self._summarize_content(full_content, sheet_name)

                    # Create output paragraph with only keep fields
                    if output_parts:
                        paragraph = "\n".join(output_parts)
                        if summary:
                            paragraph += f"\n[总结] {summary}"
                        paragraphs.append(paragraph)

                        # Update processed text in cache
                        content_hash = self._get_content_hash(full_content)
                        cache_key = self._get_cache_key(sheet_name, content_hash)
                        if cache_key in self.cache:
                            self.cache[cache_key]['processed_text'] = paragraph
                            self._save_cache()

            # Join paragraphs with double newlines
            result = "\n\n".join(paragraphs)
            logger.info(
                f"Generated {len(paragraphs)} paragraphs for sheet '{sheet_name}'"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing sheet '{sheet_name}': {str(e)}")
            raise

    def process_sheet_vertical(self, sheet_name: str) -> str:
        """
        Process a single sheet in vertical mode.

        In vertical mode:
        - Each column represents a section
        - Column headers become section titles
        - All non-empty values in the column become section content

        Args:
            sheet_name: Name of the sheet to process

        Returns:
            Formatted text content
        """
        logger.info(f"Processing sheet '{sheet_name}' in vertical mode")

        try:
            # Read specific sheet
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            logger.debug(
                f"Sheet '{sheet_name}' loaded with {len(df)} rows and "
                f"{len(df.columns)} columns"
            )

            if df.empty:
                logger.warning(f"Sheet '{sheet_name}' is empty")
                return ""

            # Get headers from first row
            headers = df.columns.tolist()
            logger.debug(f"Headers for sheet '{sheet_name}': {headers}")

            # Filter headers for output if keep_fields is specified
            output_headers = headers
            if self.keep_fields:
                output_headers = [
                    h for h in headers if h in self.keep_fields
                ]
                logger.info(f"Output headers (filtered): {output_headers}")

            sections = []

            # Process each column
            for header in output_headers:
                section_parts = []
                section_parts.append(f"## {header}")

                # Get all non-empty values from this column
                column_values = []
                for index, row in df.iterrows():
                    value = row[header]
                    if pd.notna(value):  # Skip empty values
                        value_str = str(value).strip()
                        if value_str:
                            column_values.append(value_str)

                if column_values:
                    section_parts.extend(column_values)

                    # Create section content for summarization
                    section_content = "\n".join(column_values)
                    summary = self._summarize_content(section_content, sheet_name)

                    # Add summary if available
                    if summary:
                        section_parts.append(f"[总结] {summary}")

                    sections.append("\n".join(section_parts))

                    # Update processed text in cache
                    content_hash = self._get_content_hash(section_content)
                    cache_key = self._get_cache_key(sheet_name, content_hash)
                    if cache_key in self.cache:
                        # No summary for vertical mode
                        self.cache[cache_key]['processed_text'] = "\n".join(
                            section_parts
                        )
                        self._save_cache()

            # Join sections with double newlines
            result = "\n\n".join(sections)
            logger.info(
                f"Generated {len(sections)} sections for sheet '{sheet_name}'"
            )

            return result

        except Exception as e:
            logger.error(f"Error processing sheet '{sheet_name}': {str(e)}")
            raise

    def _fallback_analysis_format(self, df, output_headers) -> str:
        """
        Fallback format when LLM analysis fails.

        Args:
            df: DataFrame containing the sheet data
            output_headers: Headers to include in output

        Returns:
            Formatted text content
        """
        lines = []
        for index, row in df.iterrows():
            row_parts = []
            for header, value in zip(df.columns, row):
                if pd.notna(value) and header in output_headers:
                    value_str = str(value).strip()
                    if value_str:
                        row_parts.append(f"{header}: {value_str}")
            if row_parts:
                lines.append(" | ".join(row_parts))

        return "\n".join(lines)

    def process_sheet_horizontal_analysis(self, sheet_name: str) -> str:
        """
        Process a single sheet in sheet-horizontal analysis mode.

        In sheet-horizontal analysis mode:
        - Analyzes the entire sheet with focus on row descriptions
        - Provides background context for better understanding
        - Generates one line per row with clear, descriptive text
        - Emphasizes accurate description of each row's content
        - Uses specialized prompt for horizontal (row-wise) description
        - Background information enhances understanding of data context

        Args:
            sheet_name: Name of the sheet to process

        Returns:
            Formatted text content with comprehensive row analysis
        """
        logger.info(f"Processing sheet '{sheet_name}' in sheet-horizontal analysis mode")

        try:
            # Read specific sheet
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            logger.debug(
                f"Sheet '{sheet_name}' loaded with {len(df)} rows and "
                f"{len(df.columns)} columns"
            )

            if df.empty:
                logger.warning(f"Sheet '{sheet_name}' is empty")
                return ""

            # Get headers from first row
            headers = df.columns.tolist()
            logger.debug(f"Headers for sheet '{sheet_name}': {headers}")

            # Filter headers for output if keep_fields is specified
            output_headers = headers
            if self.keep_fields:
                output_headers = [
                    h for h in headers if h in self.keep_fields
                ]
                logger.info(f"Output headers (filtered): {output_headers}")

            # Prepare content for LLM analysis
            content_parts = []

            # Add header information
            content_parts.append(f"Sheet: {sheet_name}")
            content_parts.append(f"Columns: {', '.join(output_headers)}")
            content_parts.append("Data:")

            # Add all data rows
            for index, row in df.iterrows():
                row_parts = []
                for header, value in zip(headers, row):
                    if pd.notna(value) and header in output_headers:
                        value_str = str(value).strip()
                        if value_str:
                            row_parts.append(f"{header}: {value_str}")
                if row_parts:
                    content_parts.append(" | ".join(row_parts))

            # Join all content for LLM analysis
            full_content = "\n".join(content_parts)

            # Generate analysis using LLM with horizontal focus
            analysis = self._analyze_content_horizontal(full_content, sheet_name)

            if analysis:
                return analysis
            else:
                # For sheet analysis modes, don't use fallback if LLM fails
                logger.error(
                    "LLM analysis failed for sheet-horizontal mode. "
                    "No output generated."
                )
                return ""

        except Exception as e:
            logger.error(f"Error processing sheet '{sheet_name}': {str(e)}")
            raise

    def process_sheet_vertical_analysis(self, sheet_name: str) -> str:
        """
        Process a single sheet in sheet-vertical analysis mode.

        In sheet-vertical analysis mode:
        - Analyzes the entire sheet with focus on row patterns and trends
        - Provides background context for better analysis
        - Generates one line per column or logical grouping
        - Emphasizes patterns and trends within each column across rows

        Args:
            sheet_name: Name of the sheet to process

        Returns:
            Formatted text content with comprehensive column analysis
        """
        logger.info(f"Processing sheet '{sheet_name}' in sheet-vertical analysis mode")

        try:
            # Read specific sheet
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            logger.debug(
                f"Sheet '{sheet_name}' loaded with {len(df)} rows and "
                f"{len(df.columns)} columns"
            )

            if df.empty:
                logger.warning(f"Sheet '{sheet_name}' is empty")
                return ""

            # Get headers from first row
            headers = df.columns.tolist()
            logger.debug(f"Headers for sheet '{sheet_name}': {headers}")

            # Filter headers for output if keep_fields is specified
            output_headers = headers
            if self.keep_fields:
                output_headers = [
                    h for h in headers if h in self.keep_fields
                ]
                logger.info(f"Output headers (filtered): {output_headers}")

            # Prepare content for LLM analysis
            content_parts = []

            # Add header information
            content_parts.append(f"Sheet: {sheet_name}")
            content_parts.append(f"Columns: {', '.join(output_headers)}")
            content_parts.append("Data:")

            # Add all data rows
            for index, row in df.iterrows():
                row_parts = []
                for header, value in zip(headers, row):
                    if pd.notna(value) and header in output_headers:
                        value_str = str(value).strip()
                        if value_str:
                            row_parts.append(f"{header}: {value_str}")
                if row_parts:
                    content_parts.append(" | ".join(row_parts))

            # Join all content for LLM analysis
            full_content = "\n".join(content_parts)

            # Generate analysis using LLM with vertical focus
            analysis = self._analyze_content_vertical(full_content, sheet_name)

            if analysis:
                return analysis
            else:
                # For sheet analysis modes, don't use fallback if LLM fails
                logger.error(
                    "LLM analysis failed for sheet-vertical mode. "
                    "No output generated."
                )
                return ""

        except Exception as e:
            logger.error(f"Error processing sheet '{sheet_name}': {str(e)}")
            raise

    def _analyze_content_horizontal(self, content: str, sheet_name: str) -> str:
        """
        Analyze content using LLM with horizontal (row-wise) focus.

        Args:
            content: The content to analyze
            sheet_name: Name of the sheet for cache key

        Returns:
            Analyzed text or empty string if LLM service is not available
        """
        if not self.llm_service:
            logger.warning(
                "LLM service not available, skipping analysis"
            )
            return ""

        # Create cache key based on entire sheet content and background info
        background_info = self.background_info or "No background context"
        cache_content = f"{content}\nBackground: {background_info}"
        content_hash = self._get_content_hash(cache_content)
        cache_key = f"sheet_horizontal_{sheet_name}_{content_hash}"

        # Check if analysis exists in cache
        if cache_key in self.cache:
            cached_analysis = self.cache[cache_key].get('processed_text', '')
            if cached_analysis:
                logger.debug(
                    f"Using cached horizontal analysis for {sheet_name}"
                )
                return cached_analysis

        try:
            # Create horizontal analysis prompt with background info
            background_info = (
                self.background_info or "No specific background context provided."
            )
            analysis_prompt = self.custom_prompt or SHEET_ANALYSIS_PROMPT.format(
                background_info=background_info
            )

            messages = [
                {
                    "role": "system",
                    "content": analysis_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ]

            analysis = self.llm_service.chat(messages)
            analysis = analysis.strip()

            # Cache the result
            self.cache[cache_key] = {
                'summary': '',  # No separate summary for analysis mode
                'content_hash': content_hash,
                'sheet_name': sheet_name,
                'original_content': content,
                'processed_text': analysis,
                'background_info': background_info,
                'mode': 'sheet_horizontal'
            }
            self._save_cache()

            logger.debug(
                f"Generated and cached horizontal analysis for {sheet_name}"
            )
            return analysis

        except Exception as e:
            logger.error(f"Error generating horizontal analysis: {str(e)}")
            return ""

    def _analyze_content_vertical(self, content: str, sheet_name: str) -> str:
        """
        Analyze content using LLM with vertical (column-wise) focus.

        Args:
            content: The content to analyze
            sheet_name: Name of the sheet for cache key

        Returns:
            Analyzed text or empty string if LLM service is not available
        """
        if not self.llm_service:
            logger.warning(
                "LLM service not available, skipping analysis"
            )
            return ""

        # Create cache key based on entire sheet content and background info
        background_info = self.background_info or "No background context"
        cache_content = f"{content}\nBackground: {background_info}"
        content_hash = self._get_content_hash(cache_content)
        cache_key = f"sheet_vertical_{sheet_name}_{content_hash}"

        # Check if analysis exists in cache
        if cache_key in self.cache:
            cached_analysis = self.cache[cache_key].get('processed_text', '')
            if cached_analysis:
                logger.debug(
                    f"Using cached vertical analysis for {sheet_name}"
                )
                return cached_analysis

        try:
            # Create vertical analysis prompt with background info
            background_info = (
                self.background_info or "No specific background context provided."
            )
            analysis_prompt = self.custom_prompt or SHEET_ANALYSIS_PROMPT.format(
                background_info=background_info
            )

            messages = [
                {
                    "role": "system",
                    "content": analysis_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ]

            analysis = self.llm_service.chat(messages)
            analysis = analysis.strip()

            # Cache the result
            self.cache[cache_key] = {
                'summary': '',  # No separate summary for analysis mode
                'content_hash': content_hash,
                'sheet_name': sheet_name,
                'original_content': content,
                'processed_text': analysis,
                'background_info': background_info,
                'mode': 'sheet_vertical'
            }
            self._save_cache()

            logger.debug(
                f"Generated and cached vertical analysis for {sheet_name}"
            )
            return analysis

        except Exception as e:
            logger.error(f"Error generating vertical analysis: {str(e)}")
            return ""

    def process_all_sheets(self, mode: str = 'horizontal') -> List[tuple]:
        """
        Process all sheets in the Excel file.

        Args:
            mode: Processing mode ('horizontal', 'vertical', 'sheet-horizontal',
                  or 'sheet-vertical')

        Returns:
            List of tuples containing (sheet_name, content, output_path)
        """
        sheet_names = self.get_sheet_names()
        results = []

        for sheet_name in sheet_names:
            logger.info(f"Processing sheet: {sheet_name}")

            # Process sheet based on mode
            if mode == 'horizontal':
                content = self.process_sheet_horizontal(sheet_name)
            elif mode == 'vertical':
                content = self.process_sheet_vertical(sheet_name)
            elif mode == 'sheet-horizontal':
                content = self.process_sheet_horizontal_analysis(sheet_name)
            elif mode == 'sheet-vertical':
                content = self.process_sheet_vertical_analysis(sheet_name)
            else:
                logger.warning(f"Unknown mode '{mode}', using horizontal mode")
                content = self.process_sheet_horizontal(sheet_name)

            # Generate output filename using sheet name
            base_name = self.excel_path.stem
            # Clean sheet name for filename (replace invalid characters)
            safe_sheet_name = "".join(
                c for c in sheet_name if c.isalnum() or c in (' ', '-', '_')
            ).rstrip()
            safe_sheet_name = safe_sheet_name.replace(' ', '-')
            output_filename = f"{base_name}-{safe_sheet_name}.txt"
            output_path = self.excel_path.parent / output_filename

            results.append((sheet_name, content, str(output_path)))

        return results

    def save_sheet_to_txt(self, content: str, output_path: str) -> str:
        """
        Save processed content to a text file.

        Args:
            content: The processed text content
            output_path: Output file path

        Returns:
            Path to the saved text file
        """
        output_path_obj = Path(output_path)

        try:
            # Ensure output directory exists
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Write content to file
            with open(output_path_obj, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"Content saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving text file: {str(e)}")
            raise

    def upload_to_kb(self, sheet_name: str, file_path: str,
                    kb_client: KBClient) -> bool:
        """
        Upload processed content to Dify knowledge base.

        Args:
            sheet_name: Name of the sheet being uploaded
            file_path: Path to the saved text file
            kb_client: KBClient instance for uploading

        Returns:
            True if upload successful, False otherwise
        """
        try:
            # Generate document name
            base_name = self.excel_path.stem
            safe_sheet_name = "".join(
                c for c in sheet_name if c.isalnum() or c in (' ', '-', '_')
            ).rstrip()
            safe_sheet_name = safe_sheet_name.replace(' ', '-')
            doc_name = f"{base_name}-{safe_sheet_name}"

            # Upload to knowledge base with hierarchical model settings
            # Parent segments: 4000 tokens, Child segments: 2000 tokens
            # Recall mode: subchunk (child segments only)
            kb_client.create_or_update_document(
                document_name=doc_name,
                file_path=file_path,
                doc_form="hierarchical_model",
                parent_separator="\n\n",
                parent_max_tokens=4000,
                parent_chunk_overlap=50,
                parent_mode="paragraph",  # Only return child segments
                subchunk_separator="\n",
                subchunk_max_tokens=2000,
                subchunk_chunk_overlap=0,
                metadata={
                    "title": f"Excel Sheet: {sheet_name}",
                    "background": f"Content from Excel file: {self.excel_path.name}",
                    "document_summary": (
                        f"Processed content from sheet '{sheet_name}'"
                    )
                }
            )

            logger.info(
                f"Successfully uploaded sheet '{sheet_name}' to knowledge base"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error uploading sheet '{sheet_name}' to KB: {str(e)}"
            )
            return False