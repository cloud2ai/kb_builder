import re
import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from devtoolbox.llm.azure_openai_provider import AzureOpenAIConfig
from devtoolbox.llm.service import LLMService
from markdown_it import MarkdownIt

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
You are a professional technical writer with deep expertise in OneProCloud's
HyperMotion and HyperBDR—host-level cloud-native solutions for migration and
disaster recovery (DR).

These solutions support:

* Production site: Includes physical servers, virtual machines, HCI,
  public/private cloud, and host-level applications such as databases
  (supporting rehost/lift-and-shift scenarios)
* Agent-based mode: Supports Linux Agent and Windows Agent for
  host-level data capture
* Agentless mode: Supports VMware, OpenStack (Ceph), AWS,
  Oracle Cloud, and Huawei FusionCompute
* Data sync:
  * Block-level differential replication
  * Periodic synchronization to cloud-native block and object storage
* Target site: Enables one-click VM recovery via cloud orchestration on
  AWS, Google Cloud, Azure, Huawei Cloud, Tencent Cloud,
  Alibaba Cloud, and others
* Recovery objectives: Minimum RPO of 5 minutes, up to 128 snapshots

---

Task

Generate a concise technical summary (max 200 characters) of a product document
segment.

The summary must:

* Preserve the original technical meaning and context
* Use relevant keywords from the source
* Focus on functionality, architecture, or usage
* Be clear, accurate, and RAG-ready
* Reflect the context of HyperMotion and HyperBDR when applicable

---

Output Format

Return a single line, with:

Title: Summary

Where:

* Title: A concise, descriptive heading using technical keywords from
  the content
* Summary: A concise description, max 200 characters, with no
  additional explanation, formatting, or line breaks
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
        self.md = MarkdownIt()

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

    def _create_section(self, title, level, path, is_root=False):
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
            'index': 1  # Initialize index to 1
        }
        if is_root:
            section['path'] = path
        return section

    def parse_markdown(self) -> Dict:
        """Parse markdown content into a structured tree

        Uses markdown-it to parse markdown text into a stream of tokens, where each
        token represents an element in the document (headings, text, etc). Then
        processes these tokens to build a section tree.

        Parsing process:
        1. markdown-it converts text into token stream
        2. Each heading creates a new section node
        3. Text between headings becomes section content
        4. Uses stack to maintain section hierarchy

        Returns:
            Dict containing the parsed markdown structure with sections and content
        """
        # Parse markdown content into tokens
        # Each token represents a specific element in the document
        # Token types include: heading_open, inline, paragraph_open, etc.
        tokens = self.md.parse(self.content)

        # Initialize variables for section tree construction
        # Root section will be the first h1 heading
        root_section = None

        # Current section being processed
        current_section = None

        # Stack to maintain section hierarchy
        section_stack = []

        # Temporary storage for section content
        current_content = []

        # Track section counts at each level
        level_counts = {}

        # Process each token in sequence
        for i, token in enumerate(tokens):
            # Handle heading tokens
            # Each heading consists of two tokens:
            # 1. heading_open: contains level information
            # 2. inline: contains the heading text
            if token.type == 'heading_open':
                # Extract heading level from tag
                level = int(token.tag[1])

                # Get heading text from next token
                heading_text = tokens[i + 1].content

                # Save accumulated content to previous section
                if current_section and current_content:
                    current_section['content'] = current_content
                    current_content = []

                # Update level count
                level_counts[level] = level_counts.get(level, 0) + 1

                # Create new section for this heading
                new_section = self._create_section(
                    heading_text, level, self.markdown_path, False
                )
                new_section['index'] = level_counts[level]

                # Handle first h1 as root section
                if level == 1 and root_section is None:
                    root_section = new_section
                    current_section = root_section
                    section_stack = [root_section]
                else:
                    # Find appropriate parent section
                    # Pop sections until finding a parent with lower level
                    while section_stack and section_stack[-1]['level'] >= level:
                        section_stack.pop()

                    if section_stack:
                        # Add new section as child of parent
                        section_stack[-1]['subsections'].append(new_section)
                        section_stack.append(new_section)
                        current_section = new_section

            # Handle text content
            # Skip heading text as it's already in section title
            elif token.type == 'inline' and current_section:
                if i > 0 and tokens[i-1].type == 'heading_open':
                    continue
                current_content.append(token.content)

        # Save content for last section if any
        if current_section and current_content:
            current_section['content'] = current_content

        # Create default root if no h1 found
        if not root_section:
            root_section = self._create_section(
                "Root", 1, self.markdown_path, True
            )
            root_section['index'] = 1

        # Save parsed structure to metadata
        self._save_to_metadata(root_section)

        logger.info(
            f"Successfully parsed document: {root_section['title']} "
            f"(sections: {len(root_section['subsections'])})"
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
            heading_text = f"{heading} [Order: {section['index']}] {section['title']}"
            logger.info(f"Writing heading: {heading_text}")
            content.append(heading_text)

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

        sections = self.parse_markdown()

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
            contents.append(f"# [{self.menu_name}]{section['title']}")
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
        logger.info(f"Saving rich markdown to {self.rich_markdown_path}")
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
