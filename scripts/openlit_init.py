"""
openlit_init.py - Early OpenLit initialization for instrumentation.

This module MUST be imported before any qdrant_client, openai, or other
instrumented libraries to ensure OpenLit can properly wrap their classes.

Usage:
    import scripts.openlit_init  # Import early in entrypoint
    
Or in __init__.py:
    from scripts import openlit_init  # Ensure early load
"""
import os

_OPENLIT_INITIALIZED = False
_OPENLIT_ENABLED = os.environ.get("OPENLIT_ENABLED", "0").lower() in ("1", "true", "yes")


def init_openlit():
    """Initialize OpenLit instrumentation. Safe to call multiple times."""
    global _OPENLIT_INITIALIZED
    
    if _OPENLIT_INITIALIZED or not _OPENLIT_ENABLED:
        return _OPENLIT_INITIALIZED
    
    try:
        import openlit
        
        _otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://openlit:4318")
        _app_name = "context-engine"
        _environment = "development"
        
        # Initialize OpenLit - this patches imported libraries
        openlit.init(
            otlp_endpoint=_otel_endpoint,
            application_name=_app_name,
            environment=_environment,
            disabled_instrumentors=None,  # Don't disable anything
            trace_content=True,  # Capture request/response content for better tracing
        )
        
        print(f"[OpenLit] Initialized with endpoint={_otel_endpoint}, app={_app_name}")
        
        # Explicitly instrument Qdrant (belt-and-suspenders for late imports)
        try:
            from openlit.instrumentation.qdrant import QdrantInstrumentor
            
            # Get tracer provider if available
            tracer_provider = None
            if hasattr(openlit, 'tracer_provider'):
                tracer_provider = openlit.tracer_provider
            
            QdrantInstrumentor().instrument(
                tracer_provider=tracer_provider,
                environment=_environment,
                application_name=_app_name,
            )
            print("[OpenLit] Qdrant instrumentation enabled")
        except ImportError:
            print("[OpenLit] Qdrant instrumentor not found (openlit may be outdated)")
        except Exception as e:
            print(f"[OpenLit] Qdrant instrumentation failed: {e}")
        
        _OPENLIT_INITIALIZED = True
        
    except ImportError:
        print("[OpenLit] SDK not installed, skipping observability")
    except Exception as e:
        print(f"[OpenLit] Failed to initialize: {e}")
    
    return _OPENLIT_INITIALIZED


# Auto-initialize on import if OPENLIT_ENABLED=1
if _OPENLIT_ENABLED:
    init_openlit()
