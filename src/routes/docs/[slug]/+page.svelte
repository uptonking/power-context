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
		color: var(--text-secondary);
		font-size: 1.1rem;
		margin-bottom: var(--spacing-lg);
		max-width: 600px;
		margin-left: auto;
		margin-right: auto;
	}

	.doc-actions {
		display: flex;
		justify-content: center;
		gap: var(--spacing-sm);
	}

	.doc-content {
		padding: var(--spacing-xl);
		line-height: 1.7;

		// Markdown content styling
		:global(h1),
		:global(h2),
		:global(h3),
		:global(h4),
		:global(h5),
		:global(h6) {
			color: var(--text-primary);
			margin-top: var(--spacing-xl);
			margin-bottom: var(--spacing-md);
			font-weight: 600;
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
			color: var(--text-secondary);
			margin-bottom: var(--spacing-md);
			font-size: 1rem;
		}

		:global(ul),
		:global(ol) {
			margin-bottom: var(--spacing-md);
			padding-left: var(--spacing-lg);
		}

		:global(li) {
			color: var(--text-secondary);
			margin-bottom: var(--spacing-xs);
		}

		:global(code) {
			background: rgba(0, 0, 0, 0.4);
			padding: 2px 6px;
			border-radius: var(--radius-xs);
			font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
			font-size: 0.9rem;
			color: #10b981;
			border: 1px solid rgba(255, 255, 255, 0.1);
		}

		:global(pre) {
			background: rgba(0, 0, 0, 0.6) !important;
			padding: var(--spacing-md);
			border-radius: var(--radius-sm);
			overflow-x: auto;
			margin: var(--spacing-md) 0;
			border: 1px solid rgba(255, 255, 255, 0.1);

			:global(code) {
				background: none !important;
				padding: 0;
				border: none;
				color: inherit;
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
			background: rgba(255, 255, 255, 0.02);
			border-radius: var(--radius-sm);
			overflow: hidden;
		}

		:global(th),
		:global(td) {
			padding: var(--spacing-sm);
			text-align: left;
			border-bottom: 1px solid rgba(255, 255, 255, 0.1);
		}

		:global(th) {
			background: rgba(255, 255, 255, 0.05);
			color: var(--text-primary);
			font-weight: 600;
		}

		:global(td) {
			color: var(--text-secondary);
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
