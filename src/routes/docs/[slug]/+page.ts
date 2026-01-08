import { error } from '@sveltejs/kit';
import { marked } from 'marked';
import hljs from 'highlight.js';

// Configure marked with syntax highlighting
marked.setOptions({
	breaks: true,
	gfm: true
});

// Import markdown files at build time
const markdownFiles = import.meta.glob('/docs/*.md', { query: '?raw', import: 'default', eager: true });

// Map of slug to document info and file path
const docMap: Record<string, { title: string; description: string; filePath: string }> = {
	'getting-started': {
		title: 'Getting Started',
		description: 'Quick start guide for Context Engine',
		filePath: '/docs/GETTING_STARTED.md'
	},
	'configuration': {
		title: 'Configuration', 
		description: 'Configure Context Engine for your needs',
		filePath: '/docs/CONFIGURATION.md'
	},
	'architecture': {
		title: 'Architecture',
		description: 'Understanding Context Engine\'s architecture',
		filePath: '/docs/ARCHITECTURE.md'
	},
	'troubleshooting': {
		title: 'Troubleshooting',
		description: 'Solve common issues and problems',
		filePath: '/docs/TROUBLESHOOTING.md'
	},
	'development': {
		title: 'Development',
		description: 'Development setup and contributing',
		filePath: '/docs/DEVELOPMENT.md'
	},
	'mcp-api': {
		title: 'MCP API',
		description: 'Model Context Protocol API reference',
		filePath: '/docs/MCP_API.md'
	}
};

export async function load({ params }) {
	const { slug } = params;
	
	if (!docMap[slug]) {
		throw error(404, 'Documentation not found');
	}

	const docInfo = docMap[slug];
	
	try {
		// Get the markdown content from the imported files
		const markdownContent = markdownFiles[docInfo.filePath] as string;
		
		if (!markdownContent) {
			throw new Error(`Markdown file not found: ${docInfo.filePath}`);
		}
		
		// Convert markdown to HTML with syntax highlighting
		const html = marked(markdownContent);
		
		return {
			slug,
			title: docInfo.title,
			description: docInfo.description,
			content: html,
			filename: docInfo.filePath.replace('/docs/', '')
		};
	} catch (err) {
		console.error(`Error processing ${slug}:`, err);
		throw error(500, 'Failed to load documentation');
	}
}