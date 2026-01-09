<script lang="ts">
	import { BookOpen, ExternalLink, FileText } from 'lucide-svelte';
	import type { PageData } from './$types';

	export let data: PageData;
</script>

<svelte:head>
	<title>{data.title} - Context Engine Documentation</title>
	<meta name="description" content={data.description} />
	<!-- Syntax highlighting CSS -->
	<link
		rel="stylesheet"
		href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css"
	/>
</svelte:head>

<div class="doc-page">
	<div class="container">
		<header class="doc-header glass">
			<div class="doc-badge">
				<FileText size={20} />
				<span>{data.filename}</span>
			</div>

			<h1 class="doc-title">{data.title}</h1>
			<p class="doc-description">{data.description}</p>

			<div class="doc-actions">
				<a
					href="https://github.com/m1rl0k/Context-Engine/blob/test/docs/{data.filename}"
					class="btn-secondary"
					target="_blank"
				>
					<ExternalLink size={16} />
					Edit on GitHub
				</a>
			</div>
		</header>

		<article class="doc-content glass">
			{@html data.content}
		</article>
	</div>
</div>

<style lang="scss">
	.doc-page {
		min-height: 100vh;
		padding: var(--spacing-lg) 0;
	}

	.container {
		max-width: 900px;
		margin: 0 auto;
		padding: 0 var(--spacing-md);
	}

	.doc-header {
		padding: var(--spacing-lg);
		margin-bottom: var(--spacing-lg);
		text-align: center;
		position: relative;
		// Enhanced readability with darker background
		background: rgba(0, 0, 0, 0.3);
		border: 1px solid rgba(255, 255, 255, 0.15);
		backdrop-filter: blur(30px);
	}

	.doc-badge {
		display: inline-flex;
		align-items: center;
		gap: var(--spacing-xs);
		padding: var(--spacing-xs) var(--spacing-sm);
		background: rgba(139, 92, 246, 0.2);
		border-radius: var(--radius-full);
		font-size: 0.85rem;
		color: var(--text-secondary);
		margin-bottom: var(--spacing-md);
		border: 1px solid rgba(139, 92, 246, 0.3);
	}

	.doc-title {
		margin-bottom: var(--spacing-sm);
		background: linear-gradient(135deg, #fff 0%, #a0a0a0 100%);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		background-clip: text;
		font-size: 2.5rem;
		font-weight: 700;
	}

	.doc-description {
		color: rgba(255, 255, 255, 0.85);
		font-size: 1.1rem;
		margin-bottom: var(--spacing-lg);
		max-width: 600px;
		margin-left: auto;
		margin-right: auto;
		text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
	}

	.doc-actions {
		display: flex;
		justify-content: center;
		gap: var(--spacing-sm);
	}

	.doc-content {
		padding: var(--spacing-xl);
		line-height: 1.7;
		// Enhanced readability with darker background
		background: rgba(0, 0, 0, 0.4);
		border: 1px solid rgba(255, 255, 255, 0.15);
		backdrop-filter: blur(30px);
		// Add scroll padding to account for navigation bar
		scroll-padding-top: 100px;

		// Markdown content styling
		:global(h1),
		:global(h2),
		:global(h3),
		:global(h4),
		:global(h5),
		:global(h6) {
			color: rgba(255, 255, 255, 0.95);
			margin-top: var(--spacing-xl);
			margin-bottom: var(--spacing-md);
			font-weight: 600;
			text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
			// Add scroll margin to headings to account for fixed nav
			scroll-margin-top: 100px;
		}

		:global(h1) {
			font-size: 2.25rem;
			border-bottom: 2px solid rgba(255, 255, 255, 0.1);
			padding-bottom: var(--spacing-sm);
		}

		:global(h2) {
			font-size: 1.875rem;
			margin-top: var(--spacing-xl);
		}

		:global(h3) {
			font-size: 1.5rem;
		}

		:global(p) {
			color: rgba(255, 255, 255, 0.85);
			margin-bottom: var(--spacing-md);
			font-size: 1rem;
			text-shadow: 0 1px 1px rgba(0, 0, 0, 0.2);
		}

		:global(ul),
		:global(ol) {
			margin-bottom: var(--spacing-md);
			padding-left: var(--spacing-lg);
		}

		:global(li) {
			color: rgba(255, 255, 255, 0.85);
			margin-bottom: var(--spacing-xs);
			text-shadow: 0 1px 1px rgba(0, 0, 0, 0.2);
		}

		:global(code) {
			background: rgba(0, 0, 0, 0.6);
			padding: 3px 8px;
			border-radius: var(--radius-xs);
			font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
			font-size: 0.9rem;
			color: #22d3ee;
			border: 1px solid rgba(34, 211, 238, 0.3);
			text-shadow: 0 1px 1px rgba(0, 0, 0, 0.5);
		}

		:global(pre) {
			background: rgba(0, 0, 0, 0.8) !important;
			padding: var(--spacing-md);
			border-radius: var(--radius-sm);
			overflow-x: auto;
			margin: var(--spacing-md) 0;
			border: 1px solid rgba(255, 255, 255, 0.2);
			box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);

			:global(code) {
				background: none !important;
				padding: 0;
				border: none;
				color: inherit;
				text-shadow: none;
			}
		}

		:global(blockquote) {
			border-left: 4px solid rgba(139, 92, 246, 0.5);
			padding-left: var(--spacing-md);
			margin: var(--spacing-md) 0;
			color: var(--text-muted);
			font-style: italic;
			background: rgba(139, 92, 246, 0.05);
			padding: var(--spacing-md);
			border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
		}

		:global(table) {
			width: 100%;
			border-collapse: collapse;
			margin: var(--spacing-md) 0;
			background: rgba(0, 0, 0, 0.4);
			border-radius: var(--radius-sm);
			overflow: hidden;
			border: 1px solid rgba(255, 255, 255, 0.15);
		}

		:global(th),
		:global(td) {
			padding: var(--spacing-sm);
			text-align: left;
			border-bottom: 1px solid rgba(255, 255, 255, 0.15);
		}

		:global(th) {
			background: rgba(0, 0, 0, 0.4);
			color: rgba(255, 255, 255, 0.95);
			font-weight: 600;
			text-shadow: 0 1px 1px rgba(0, 0, 0, 0.3);
		}

		:global(td) {
			color: rgba(255, 255, 255, 0.85);
			text-shadow: 0 1px 1px rgba(0, 0, 0, 0.2);
		}

		:global(a) {
			color: #8b5cf6;
			text-decoration: none;
			transition: color 0.2s ease;

			&:hover {
				color: #a78bfa;
				text-decoration: underline;
			}
		}

		:global(hr) {
			border: none;
			height: 1px;
			background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
			margin: var(--spacing-xl) 0;
		}

		// Syntax highlighting adjustments
		:global(.hljs) {
			background: rgba(0, 0, 0, 0.6) !important;
			color: #e6edf3 !important;
		}
	}

	@media (max-width: 768px) {
		.doc-title {
			font-size: 2rem;
		}

		.doc-content {
			padding: var(--spacing-lg);
		}
	}
</style>
