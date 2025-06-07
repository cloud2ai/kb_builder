import re
import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from devtoolbox.llm.azure_openai_provider import AzureOpenAIConfig
from devtoolbox.llm.service import LLMService

# Initialize logger
logger = logging.getLogger(__name__)

DEFAULT_HEADING_LEVEL = 2

PROCESS_TABLE_PROMPT = """
You are a professional solutions engineer.

Your task: Convert all tables in the given Markdown content into natural
language descriptions, while preserving the surrounding non-table content
unchanged. Output the final result as plain text, without any additional
explanation.

Instructions:
- For every table in the Markdown:
  - Convert each row into a natural language sentence that clearly
    describes its function or purpose.
  - If a cell contains a link, keep the link in the sentence.
  - Use the background and context outside the table to guide how the
    information should be described.
- Do not change or summarize any content that is outside the tables.
  Keep it as-is.
- Ensure the output language remains the same as the input (e.g., if the
  Markdown is in Chinese, the output should also be in Chinese).
- Return the entire updated Markdown, replacing each table with the
  natural language version, and preserving everything else.
"""

SUMMARY_PROMPT = """
You are a professional technical writer with knowledge of OneProCloud's
HyperMotion and HyperBDR—cloud-native solutions for migration and disaster
recovery. These solutions support agent-based and agentless modes, perform
block-level differential replication, periodically synchronize data to
cloud-native block and object storage, and enable one-click VM recovery via
cloud orchestration on platforms like AWS, Google Cloud, Azure, and Huawei
Cloud. The minimum RPO is 5 minutes, with support for up to 128 snapshots.

Generate a concise summary (maximum 200 characters) of a product technical
document. The summary must:

Accurately preserve the original meaning and technical context

Use key terms and phrases from the source text whenever possible

Focus on the core functionality, architecture, and usage details

Be technically accurate, clear, and suitable for Retrieval-Augmented
Generation (RAG)

Reflect the context of HyperMotion and HyperBDR whenever relevant

Return only the summary in a single paragraph, with no explanation or
formatting.
"""

SUMMARY_QA_PROMPT = """
You are a professional technical content generator working with product
knowledge of OneProCloud's HyperMotion and HyperBDR — cloud-native
solutions for migration and disaster recovery. These products support both
agent-based and agentless modes, block-level differential replication,
periodic data sync to block and object storage, fast RPO (5 minutes), up to
128 snapshots, and one-click VM recovery through cloud orchestration on
platforms like AWS, Google Cloud, Azure, and Huawei Cloud.

Given a technical paragraph, generate 10 precise and relevant questions
that could be asked based on its content. The questions should:

- Reflect the context of HyperMotion and HyperBDR's architecture and
  capabilities
- Use terminology or key phrases from the paragraph when possible
- Address aspects such as use cases, features, benefits, technical
  mechanisms, and performance metrics
- Be concise, clear, and one question per line
- Be suitable for RAG, FAQ generation, or knowledge extraction

Return exactly 10 questions, one per line, without explanations or extra
text.
"""


class RichMarkdown:
    """
    Represents a rich markdown document with its structure and metadata.

    This class handles the parsing and processing of markdown documents,
    including caching, content conversion, and summary generation.
    """
    def __init__(
        self,
        markdown_path: str,
        converted_dir: str,
        converted_filename: str,
        menu_name: str = None,
        summary_level: int = 2,
        converted_level: int = 3
    ):
        self.markdown_path = markdown_path
        self.converted_dir = converted_dir
        self.converted_filename = converted_filename
        self.menu_name = menu_name
        self.summary_level = summary_level
        self.converted_level = converted_level

        if markdown_path:
            try:
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    self.content = f.read()
                logger.debug(
                    f"Loaded markdown content from {markdown_path}"
                )
            except Exception as e:
                logger.error(
                    f"Error reading markdown file {markdown_path}: {e}"
                )
                raise
        else:
            logger.error("markdown_path must be provided.")
            raise ValueError("markdown_path must be provided.")

        self.llm_service = self._init_llm()

        # Load metadata and merge with sections
        self.metadata = self._load_metadata()
        self.sections = self._merge_sections_and_metadata()

    def _init_llm(self):
        # Initialize LLM service
        openai_config = AzureOpenAIConfig(
            temperature=0.1,
            max_tokens=10000
        )

        logger.debug(
            "LLM service initialized with config: %s",
            openai_config
        )
        return LLMService(openai_config)

    @property
    def converted_path(self) -> Path:
        """Get the path for the converted file."""
        return Path(self.converted_dir) / self.converted_filename

    @property
    def metadata_path(self) -> Path:
        """Get the path for the metadata file."""
        return (
            Path(self.converted_dir) /
            f"{self.converted_filename}.metadata.json"
        )

    @property
    def rich_markdown_path(self) -> Path:
        """Get the path for the rich markdown file."""
        return (
            Path(self.converted_dir) /
            f"{self.converted_filename}.txt"
        )

    def _summarize_text(
        self,
        text: str,
        prompt: str = None
    ) -> str:
        """
        Summarize text using LLM.

        Args:
            text: The text to summarize
            prompt: Optional custom prompt for summarization

        Returns:
            The summarized text
        """
        messages = [
            {
                "role": "system",
                "content": prompt if prompt else SUMMARY_PROMPT
            },
            {
                "role": "user",
                "content": text
            }
        ]
        return self.llm_service.chat(messages)

    def _convert_content(self, content: str) -> str:
        """Convert tables in the content to plain text."""
        # Convert background content
        if self._contains_table(content):
            return self._process_table(content)
        else:
            return content

    def _contains_table(self, markdown_text: str) -> bool:
        """Check if the markdown content contains a table."""
        table_regex = r'((?:\| *[^|\r\n]+ *)+\|)'
        return bool(re.search(table_regex, markdown_text, re.DOTALL))

    def _process_table(self, content: str) -> str:
        """Process the table in the markdown content."""
        messages = [
            {
                "role": "system",
                "content": PROCESS_TABLE_PROMPT
            },
            {
                "role": "user",
                "content": content
            }
        ]
        return self.llm_service.chat(messages)

    def _save_to_metadata(self, sections=None):
        """
        Save current state to metadata file.

        Args:
            sections: The sections to save. If None, use self.sections
        """
        try:
            sections_to_save = sections or self.sections
            logger.debug(
                f"\n{'='*50}\n"
                f"Saving metadata to {self.metadata_path}\n"
                f"Root section fields: {list(sections_to_save.keys())}\n"
                f"Root section values:\n"
                f"  summary: {sections_to_save.get('summary', 'Not found')}\n"
                f"  converted_content: "
                f"{sections_to_save.get('converted_content', 'Not found')}\n"
                f"Subsections count: "
                f"{len(sections_to_save.get('subsections', []))}\n"
                f"{'='*50}"
            )

            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(sections_to_save, f, ensure_ascii=False, indent=2)
            logger.info(
                f"Successfully saved metadata to {self.metadata_path} "
                f"(sections: {len(sections_to_save)})"
            )
        except Exception as e:
            logger.error(
                f"Error saving metadata to {self.metadata_path}: {str(e)}"
            )

    def _load_metadata(self) -> dict:
        """
        Load metadata for the markdown content.

        Returns:
            dict: The loaded metadata
        """
        metadata_path = str(self.metadata_path)
        if not os.path.exists(metadata_path):
            logger.info(f"No metadata file found at {metadata_path}")
            return {}

        logger.info(f"Attempting to load metadata from: {metadata_path}")

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(
                f"\n{'='*50}\n"
                f"Successfully loaded metadata from {metadata_path}\n"
                f"Root section fields: {list(metadata.keys())}\n"
                f"Root section values:\n"
                f"  summary: {metadata.get('summary', 'Not found')}\n"
                f"  converted_content: "
                f"{metadata.get('converted_content', 'Not found')}\n"
                f"Subsections count: {len(metadata.get('subsections', []))}\n"
                f"{'='*50}"
            )
            logger.debug(f"RawMetadata: {metadata}")
            return metadata
        except Exception as e:
            logger.warning(
                f"Error loading metadata from {metadata_path}: {str(e)}"
            )
            return {}

    def parse_sections(self):
        """
        Parse markdown content into a rich markdown structure.

        The document must have:
        1. A single level 1 heading (#) as the document title
        2. Optional level 2 headings (##) for sections

        Content between level 1 heading and first level 2 heading
        (or end of file) will be treated as background information.

        The parsed structure includes:
        - Section hierarchy with title, content, and subsections
        - Content length statistics for each section and level
        - Maximum depth information for each branch
        - Document metadata including total sections and content length

        Returns:
            dict: The root section with metadata
        """
        logger.info(f"Starting markdown parsing for file: {self.markdown_path}")
        logger.debug(f"Menu name: {self.menu_name}")

        # Initialize parsing state
        # Store all root sections
        sections = []
        # Track current section hierarchy for building the tree structure
        section_stack = []
        # Flag for code block state to avoid processing headings inside code
        # blocks
        in_code_block = False
        # Track document structure metadata
        metadata = {
            'max_depth': 1,  # Maximum heading level in the document
            'total_sections': 0,  # Total number of sections
            'level_counts': {1: 0},  # Count of sections at each level
            'level_content_lengths': {1: 0},  # Total content length per level
            'total_content_length': 0  # Total content length of the document
        }

        logger.info("Starting markdown parsing with hierarchical structure")
        logger.debug(f"Initial state: in_code_block={in_code_block}")

        # Create standardized section structure
        def create_section(title, level, path, is_root=False):
            """
            Create a standardized section structure.

            Each section contains:
            - title: Section heading text
            - content: List of content lines
            - subsections: List of child sections
            - level: Heading level (number of #)
            - content_length: Length of content in this section
            - branch_max_depth: Maximum depth in this section's branch
            - index: The index of this section in its parent's subsections (starts from 1)

            Args:
                title: Section title
                level: Heading level (number of #)
                path: Document path (only needed for root sections)
                is_root: Whether this is a root section

            Returns:
                Dictionary containing complete section information
            """
            section = {
                'title': title,
                'content': [],
                'subsections': [],
                'level': level,
                'summary': "",
                'converted_content': None,
                'content_length': 0,  # Track content length for this section
                'branch_max_depth': level,  # Track max depth in this branch
                'index': 1  # Initialize index to 1
            }
            if is_root:
                section['path'] = path
            return section

        def update_branch_max_depth(section):
            """Update max depth information for a section branch.

            This function recursively traverses the section tree to:
            1. Calculate the maximum depth in each branch
            2. Update the branch_max_depth field for each section
            3. Return the maximum depth found in the branch

            Args:
                section: The section to update

            Returns:
                The maximum depth in this branch
            """
            max_depth = section['level']
            if 'subsections' in section:
                for subsection in section['subsections']:
                    subsection_depth = update_branch_max_depth(subsection)
                    max_depth = max(max_depth, subsection_depth)
            section['branch_max_depth'] = max_depth
            return max_depth

        # Split content into lines for processing
        lines = self.content.splitlines()
        logger.debug(f"Total lines in document: {len(lines)}")

        # Skip frontmatter if exists (content between --- markers)
        start_idx = 0
        if lines and lines[0].strip() == '---':
            logger.debug("Found frontmatter, skipping...")
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == '---':
                    start_idx = i + 1
                    logger.debug(f"Frontmatter ends at line {start_idx}")
                    break

        # Process each line to build the section hierarchy
        for line_num, line in enumerate(lines[start_idx:], start_idx):
            # Skip empty lines
            line = line.strip()
            if not line:
                continue

            # Skip table of contents marker
            if line.strip() == "[[toc]]":
                logger.debug(f"Found TOC marker at line {line_num}")
                continue

            # Handle code blocks to avoid processing headings inside them
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                logger.debug(
                    f"Code block {'started' if in_code_block else 'ended'} "
                    f"at line {line_num}"
                )
                # Add code block marker to content
                if not section_stack:
                    # Create root section if none exists
                    if not sections:
                        root_section = create_section(
                            "Root", 0, self.markdown_path, True)
                        sections.append(root_section)
                        section_stack.append(root_section)
                section_stack[-1]['content'].append(line)
                section_stack[-1]['content_length'] += len(line)
                metadata['total_content_length'] += len(line)
                continue

            # Process headings only outside code blocks
            if not in_code_block and line.startswith('#'):
                level = line.count('#')
                heading_text = line[level:].strip()
                logger.debug(
                    f"Found heading at line {line_num}: "
                    f"level {level}, text: {heading_text}"
                )

                # Update metadata
                metadata['max_depth'] = max(metadata['max_depth'], level)
                metadata['total_sections'] += 1
                metadata['level_counts'][level] = (
                    metadata['level_counts'].get(level, 0) + 1
                )

                # Create new section
                is_root = not section_stack
                new_section = create_section(
                    heading_text, level, self.markdown_path, is_root
                )

                # Handle section hierarchy based on heading level
                if not section_stack:
                    # First section, add to root
                    sections.append(new_section)
                    section_stack.append(new_section)
                else:
                    current_level = section_stack[-1]['level']
                    if level > current_level:
                        # New section is a subsection of current section
                        new_section['index'] = (
                            len(section_stack[-1]['subsections']) + 1
                        )
                        section_stack[-1]['subsections'].append(new_section)
                        section_stack.append(new_section)
                    elif level == current_level:
                        # New section is at same level as current section
                        section_stack.pop()
                        if section_stack:
                            new_section['index'] = (
                                len(section_stack[-1]['subsections']) + 1
                            )
                            section_stack[-1]['subsections'].append(new_section)
                        else:
                            new_section['index'] = len(sections) + 1
                            sections.append(new_section)
                    else:
                        # New section is at higher level
                        while (section_stack and
                               section_stack[-1]['level'] >= level):
                            section_stack.pop()
                        if section_stack:
                            new_section['index'] = (
                                len(section_stack[-1]['subsections']) + 1
                            )
                            section_stack[-1]['subsections'].append(new_section)
                        else:
                            new_section['index'] = len(sections) + 1
                            sections.append(new_section)
                        section_stack.append(new_section)
            else:
                # Process non-heading content
                if not section_stack:
                    # Create root section if none exists
                    if not sections:
                        root_section = create_section(
                            "Root", 0, self.markdown_path, True)
                        sections.append(root_section)
                        section_stack.append(root_section)
                section_stack[-1]['content'].append(line)
                section_stack[-1]['content_length'] += len(line)
                metadata['total_content_length'] += len(line)
                # Update level content length
                current_level = section_stack[-1]['level']
                metadata['level_content_lengths'][current_level] = (
                    metadata['level_content_lengths'].get(current_level, 0) +
                    len(line)
                )

        # Get the root section
        root_section = sections[0]
        # Update branch max depth for all sections
        update_branch_max_depth(root_section)
        # Add metadata to root section
        root_section['metadata'] = metadata

        # Save the parsed sections to metadata
        self._save_to_metadata(root_section)

        logger.info(
            f"Successfully parsed document: {root_section['title']} "
            f"(sections: {metadata['total_sections']})"
        )

        return root_section

    def _get_all_content(self, section):
        """Get all content including subsections.

        Args:
            section: The section dictionary containing content and subsections

        Returns:
            List of content strings from the section and all its subsections
        """
        content = []
        # Add current section's content
        if isinstance(section['content'], list):
            content.extend(section['content'])
        else:
            content.append(section['content'])

        # Add subsections' content recursively
        if 'subsections' in section:
            for subsection in section['subsections']:
                content.extend(self._get_all_content(subsection))

        return content

    def _get_sections_by_level(self, section):
        """Get all sections organized by level.

        Args:
            section: The root section to process

        Returns:
            dict: Sections organized by level
        """
        sections_by_level = {}
        self._collect_sections_by_level(section, sections_by_level)
        return sections_by_level

    def _collect_sections_by_level(self, section, sections_by_level):
        """Recursively collect all sections and organize them by level.

        Args:
            section: The section to process
            sections_by_level: Dictionary to store sections by level
        """
        level = section['level']
        if level not in sections_by_level:
            sections_by_level[level] = []
        sections_by_level[level].append(section)

        for subsection in section['subsections']:
            self._collect_sections_by_level(subsection, sections_by_level)

    def _collect_content(self, section, include_title=True, recursive=True):
        """Collect content from a section and optionally its subsections.

        Args:
            section: The section to collect content from
            include_title: Whether to include section title with proper heading level.
                         Defaults to True.
            recursive: Whether to collect content from subsections recursively.
                      Defaults to True.

        Returns:
            str: Combined content from the section and optionally its subsections
        """
        content = []

        # Add section title with proper heading level if requested
        if include_title and section['title'] != 'Root':
            heading = '#' * section['level']
            content.append(f"{heading} [Order: {section['index']}] {section['title']}")

        # Add section's own content
        if section['content']:
            content.extend(section['content'])

        # Add content from subsections if recursive is True
        if recursive:
            for subsection in section['subsections']:
                content.extend(self._collect_content(
                    subsection, include_title, recursive
                ))

        return content

    def _collect_section_summaries(self, section, target_level):
        """Collect summaries from sections at the target level.

        Args:
            section: The section to process
            target_level: The target level to collect summaries from

        Returns:
            list: List of summaries from target level sections
        """
        summaries = []

        # Process current section
        if section['level'] == target_level and section.get('summary'):
            summaries.append(section['summary'])

        # Process all subsections recursively
        if 'subsections' in section:
            for subsection in section['subsections']:
                summaries.extend(
                    self._collect_section_summaries(subsection, target_level)
                )

        return summaries

    def _process_section_summaries(self, section, current_level):
        """Process a section at the target level.

        Args:
            section: The section to process
            current_level: Current level in the document

        Returns:
            The processed section
        """
        logger.debug(
            f"\n{'*'*100}\n"
            f"Processing summary section: '{section['title']}'\n"
            f"Level: {current_level}\n"
            f"Summary Level: {self.summary_level}\n"
            f"{'*'*100}"
        )

        # Process subsections first
        for subsection in section['subsections']:
            # Stop if subsection level is greater than summary level
            subsection_level = subsection['level']
            if subsection_level > self.summary_level:
                logger.debug(f"Breaking at: '{section['title']}'")
                break

            # Continue to handle lower level subsections
            self._process_section_summaries(
                subsection, current_level + 1
            )

        # Process summary based on level
        if current_level == self.summary_level:
            # Generate summary for current level with all subsections content
            content = self._collect_content(section)
            if content and not section.get('summary'):
                content_text = '\n'.join(content)
                summary_content = f"Title: {section['title']}\n{content_text}"
                section['summary'] = self._summarize_text(summary_content)
        elif current_level < self.summary_level:
            # Generate summary for parent level using children's summaries
            summaries = self._collect_section_summaries(
                section, self.summary_level)

            # If no summaries from children use its own content
            if not summaries and section.get('content'):
                summaries = section['content']

            if summaries and not section.get('summary'):
                content_text = '\n'.join(summaries)
                section['summary'] = self._summarize_text(content_text)

        return section

    def _process_section_converted_content(
        self, section, current_level, parent_section
    ):
        """Process converted content for a section.

        Args:
            section: The section to process
            current_level: Current section level
            parent_section: Parent section
        """
        logger.debug(f"Processing converted content for section: '{section['title']}'")
        # Process subsections first
        for subsection in section['subsections']:
            if subsection['level'] > self.converted_level:
                logger.debug(
                    f"Breaking at: '{subsection['title']}' "
                    f"level: {subsection['level']}"
                )
                break

            self._process_section_converted_content(
                subsection, current_level + 1, section
            )

        # Skip if already processed
        if section.get('converted_content'):
            return

        # Collect content based on level
        if current_level < self.converted_level and not section['subsections']:
            # For leaf nodes below converted_level, only process current content
            content = self._collect_content(section, recursive=False)
            # For the second level we do not need to add summary
            content[0] = (
                f"{content[0]}\nSummary: {section['summary']}"
            )
        elif current_level == self.converted_level:
            # For converted_level, process all content including parent
            content = self._collect_content(section)
            parent_content = self._collect_content(
                parent_section, recursive=False
            )
            parent_summary = (
                f"{parent_content[0]}\nSummary: {parent_section['summary']}"
            )
            content.insert(0, parent_summary)
        else:
            return

        # Convert and cache the content
        content_text = '\n'.join(content)
        logger.debug(
            f"Converting content for section: '{section['title']}'"
        )
        section['converted_content'] = self._convert_content(content_text)

    def generate_rich_markdown(self):
        """Generate rich markdown with summaries.

        This method generates summaries for sections at the specified level.
        For each section at the target level:
        - Collect all content from its subsections
        - Generate a single summary for the entire content

        Returns:
            dict: The root section with generated summaries
        """
        logger.info(
            f"\n{'#'*100}\n"
            f"Starting summary generation\n"
            f"Target summary level: {self.summary_level}\n"
            f"Target converted level: {self.converted_level}\n"
            f"{'#'*100}"
        )

        # Process the root section
        self._process_section_summaries(self.sections, 1)
        logger.info("Starting converted content generation")
        self._process_section_converted_content(self.sections, 1, None)
        logger.info("Converted content generation completed")

        # Generate document-level summary from level 2 summaries
        logger.info(
            f"\n{'#'*100}\n"
            f"Generating document-level summary\n"
            f"{'#'*100}"
        )

        # Collect summaries from specific level
        summaries = self._collect_section_summaries(
            self.sections, self.summary_level - 1
        )
        if summaries:
            content_text = '\n'.join(summaries)

            # Check if we have a cached summary in metadata
            if self.sections.get('summary'):
                logger.debug("Using cached summary")
            else:
                self.sections['summary'] = self._summarize_text(content_text)

        logger.info(
            f"\n{'#'*100}\n"
            f"Summary generation completed\n"
            f"{'#'*100}"
        )

        # Save the processed content to metadata
        self._save_to_metadata()

        return self.sections

    def _get_section_content(self, section):
        """Get the content of a section for comparison."""
        content = []
        if section['title'] != 'Root':
            content.append(f"{'#' * section['level']} {section['title']}")
        content.extend(section['content'])
        return '\n'.join(content)

    def _merge_section(self, section, cached_section):
        """Merge a section with its cached version if content matches."""
        if not cached_section:
            logger.debug(
                f"No cached section found for: '{section['title']}'"
            )
            return section

        # Compare content
        current_content = self._get_section_content(section)
        cached_content = self._get_section_content(cached_section)

        logger.debug(
            f"\n{'='*50}\n"
            f"Comparing section: '{section['title']}'\n"
            f"Current content length: {len(current_content)}\n"
            f"Cached content length: {len(cached_content)}\n"
            f"Content match: {current_content == cached_content}\n"
            f"Current content preview: {current_content[:100]}\n"
            f"Cached content preview: {cached_content[:100]}\n"
            f"{'='*50}"
        )

        # Create a new section to avoid modifying the original
        merged_section = section.copy()

        if current_content == cached_content:
            # Content matches, merge cached fields
            logger.debug(
                f"\n{'='*50}\n"
                f"Content matches for section: '{section['title']}'\n"
                f"Available cached fields: {list(cached_section.keys())}\n"
                f"Cached values:\n"
                f"  summary: {cached_section.get('summary', 'Not found')}\n"
                f"  converted_content: {cached_section.get('converted_content', 'Not found')}\n"
                f"Raw cached section: {cached_section}\n"
                f"{'='*50}"
            )

            # Merge each cached field
            for cache_field in self.CACHE_FIELDS:
                if cache_field in cached_section:
                    cached_value = cached_section[cache_field]
                    logger.debug(
                        f"\n{'-'*30}\n"
                        f"Merging field '{cache_field}' for section: "
                        f"'{section['title']}'\n"
                        f"Current value: {merged_section.get(cache_field)}\n"
                        f"Cached value: {cached_value}\n"
                        f"Value type: {type(cached_value)}\n"
                        f"{'-'*30}"
                    )
                    merged_section[cache_field] = cached_value
        else:
            logger.debug(
                f"\n{'='*50}\n"
                f"Content changed for section: '{section['title']}'\n"
                f"First 100 chars of current content:\n"
                f"{current_content[:100]}\n"
                f"First 100 chars of cached content:\n"
                f"{cached_content[:100]}\n"
                f"{'='*50}"
            )

        # Process subsections
        merged_section['subsections'] = []
        for subsection in section['subsections']:
            cached_subsection = next(
                (s for s in cached_section.get('subsections', [])
                 if s['title'] == subsection['title']),
                None
            )
            if cached_subsection:
                logger.debug(
                    f"\n{'-'*30}\n"
                    f"Found cached subsection: '{subsection['title']}'\n"
                    f"Cached fields: {list(cached_subsection.keys())}\n"
                    f"Raw cached subsection: {cached_subsection}\n"
                    f"{'-'*30}"
                )
            merged_subsection = self._merge_section(
                subsection, cached_subsection
            )
            merged_section['subsections'].append(merged_subsection)

        return merged_section

    def _merge_sections_and_metadata(self):
        """Merge sections with metadata, ensuring content consistency.

        Returns:
            The merged sections with cached fields if content matches
        """
        # Fields that should be merged from cache if content matches
        self.CACHE_FIELDS = {
            'summary',  # Summary of the section
            'converted_content',  # Converted content (e.g., tables converted to text)
            # Add more fields here as needed
        }

        logger.info(
            f"\n{'#'*50}\n"
            f"Starting metadata merge\n"
            f"Cache fields to merge: {self.CACHE_FIELDS}\n"
            f"{'#'*50}"
        )

        sections = self.parse_sections()

        # Start merging from root
        return self._merge_section(sections, self.metadata)

    def _process_section_for_rich_markdown(self, section: dict) -> list:
        """Process a section and its subsections for rich markdown output.

        Args:
            section: The section to process

        Returns:
            List of content lines for this section
        """
        contents = []

        # Add document title and summary for root level
        if section['level'] == 1:
            contents.append(f"# {self.menu_name}")
            if section.get('summary'):
                contents.append(f"Summary: {section['summary']}")
            contents.append("")

        # Add converted content if exists
        if section.get('converted_content'):
            contents.append(
                self._remove_empty_lines(section['converted_content'])
            )
            contents.append("")

        # Process subsections
        for subsection in section['subsections']:
            subsection_contents = self._process_section_for_rich_markdown(
                subsection
            )
            if subsection_contents:
                contents.extend(subsection_contents)

        return contents

    def _remove_empty_lines(self, content: str) -> str:
        """Remove all empty lines from content.

        This method processes the input content by:
        1. Splitting it into lines
        2. Filtering out empty lines
        3. Joining the remaining lines back together

        Args:
            content: Input content string

        Returns:
            Content string with all empty lines removed
        """
        if not content:
            return content

        # Process content by splitting into lines and filtering empty ones
        lines = content.splitlines()
        return '\n'.join(line for line in lines if line.strip())

    def save_rich_markdown(self):
        """Save rich markdown content with hierarchical structure.

        The output format will be:
        1. Document title and summary
        2. For each section at or above min_level:
           - Section title
           - Section summary
           - Section content (including subsections below min_level)
           - Section number (if applicable)
        """
        # Process all sections
        contents = self._process_section_for_rich_markdown(self.sections)

        # Save to file
        try:
            with open(self.rich_markdown_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(contents))
            logger.info(
                f"Successfully saved rich markdown to "
                f"{self.rich_markdown_path}"
            )
        except Exception as e:
            logger.error(
                f"Error saving rich markdown to {self.rich_markdown_path}: "
                f"{str(e)}"
            )
            raise
