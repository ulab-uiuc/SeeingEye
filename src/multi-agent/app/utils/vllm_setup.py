"""
vLLM Setup and Management Utilities

Consolidated vLLM utilities for cluster port forwarding, server management,
and inference setup for multi-agent workflows.

This module combines functionality from the original vllm_utils.py
"""

import subprocess
import time
import requests
import signal
import os
import atexit
import math
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Import logger - handle case where it might not be available in utils context
try:
    from app.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class VLLMPortForwarding:
    """Manages SSH port forwarding for vLLM servers in cluster environment"""

    def __init__(self):
        self.forwarding_pids = []
        self.server_configs = {}
        self.setup_complete = False

        # Register cleanup on exit
        atexit.register(self.cleanup)

    def cleanup(self):
        """Clean up port forwarding processes"""
        if self.forwarding_pids:
            logger.info("🧹 Cleaning up vLLM port forwarding...")
            for pid in self.forwarding_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                    logger.debug(f"Stopped port forwarding (PID: {pid})")
                except:
                    pass

            # Kill any remaining SSH port forwarding
            try:
                subprocess.run(['pkill', '-f', 'ssh.*-L.*800[01]'],
                             capture_output=True, check=False)
            except:
                pass

            self.forwarding_pids = []
            self.server_configs = {}
            self.setup_complete = False

    def get_gpu_nodes(self) -> List[str]:
        """Get list of GPU nodes for current user"""
        try:
            result = subprocess.run(['squeue', '-u', os.getenv('USER'), '-h', '-o', '%N'],
                                  capture_output=True, text=True, check=True)
            nodes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    nodes.extend(line.strip().split(','))
            return list(set(nodes))  # Remove duplicates
        except Exception as e:
            logger.warning(f"Could not get GPU nodes: {e}")
            return []

    def check_vllm_on_node(self, node: str, vision_port: int = 8000, text_port: int = 8001) -> Dict[str, Dict]:
        """Check for vLLM processes on a specific node for specific ports"""
        configs = {}
        try:
            # Check for vLLM processes
            result = subprocess.run([
                'ssh', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=no',
                node, 'ps aux | grep -v grep | grep vllm'
            ], capture_output=True, text=True, timeout=15)

            if result.returncode == 0 and result.stdout.strip():
                # Check for the specific ports only
                port_result = subprocess.run([
                    'ssh', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=no',
                    node, f'ss -tulpn | grep -E "(:{vision_port}|:{text_port})"'
                ], capture_output=True, text=True, timeout=15)

                if port_result.returncode == 0 and port_result.stdout.strip():
                    ports_info = port_result.stdout.strip().split('\n')

                    for line in ports_info:
                        if f':{vision_port}' in line:
                            configs['vision'] = {
                                'node': node,
                                'port': vision_port,
                                'local_port': vision_port,
                                'model': 'Qwen/Qwen2.5-VL-3B-Instruct',
                                'name': f'Vision Model (Qwen2.5-VL-3B) - Port {vision_port}',
                                'config_name': 'translator_api'
                            }
                        elif f':{text_port}' in line:
                            configs['text'] = {
                                'node': node,
                                'port': text_port,
                                'local_port': text_port,
                                'model': 'Qwen/Qwen3-8B',
                                'name': f'Text Model (Qwen3-8B) - Port {text_port}',
                                'config_name': 'reasoning_api'
                            }
        except Exception as e:
            logger.debug(f"Error checking {node}: {e}")

        return configs

    def discover_servers(self, vision_port: int = 8000, text_port: int = 8001) -> bool:
        """Discover vLLM servers across GPU nodes for specific ports"""
        gpu_nodes = self.get_gpu_nodes()
        if not gpu_nodes:
            logger.warning("No GPU nodes found for current user")
            return False

        for node in gpu_nodes:
            node_configs = self.check_vllm_on_node(node, vision_port=vision_port, text_port=text_port)
            self.server_configs.update(node_configs)

        return len(self.server_configs) > 0

    def setup_forwarding(self, server_type: str, config: Dict) -> bool:
        """Set up SSH port forwarding for a specific server"""
        node = config['node']
        remote_port = config['port']
        local_port = config.get('local_port', remote_port)

        try:
            logger.info(f"Setting up forwarding: localhost:{local_port} -> {node}:{remote_port}")

            # Start SSH port forwarding
            proc = subprocess.Popen([
                'ssh', '-N', '-L', f'{local_port}:localhost:{remote_port}',
                '-o', 'StrictHostKeyChecking=no', node
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            self.forwarding_pids.append(proc.pid)
            config['forwarding_pid'] = proc.pid

            # Wait for connection to establish
            time.sleep(2)

            # Test the forwarding
            if self.test_port_connectivity(local_port):
                logger.info(f"✅ Port {local_port} forwarding working")
                return True
            else:
                logger.warning(f"⚠️ Port {local_port} forwarding not ready yet")
                return False

        except Exception as e:
            logger.error(f"Failed to set up forwarding for {node}:{remote_port} -> localhost:{local_port} - {e}")
            return False

    def test_port_connectivity(self, port: int) -> bool:
        """Test if a specific port is accessible on localhost"""
        try:
            response = requests.get(f'http://localhost:{port}/health', timeout=3)
            return response.status_code == 200
        except:
            return False

    def test_node_connectivity(self, node: str, port: int) -> bool:
        """Test if a specific port is accessible on a remote node"""
        try:
            response = requests.get(f'http://{node}:{port}/health', timeout=5)
            return response.status_code == 200
        except:
            return False

    def setup_all_forwarding(self, vision_port: int = 8000, text_port: int = 8001) -> Dict[str, bool]:
        """Set up port forwarding for all discovered servers"""
        if not self.server_configs:
            if not self.discover_servers(vision_port=vision_port, text_port=text_port):
                logger.error("No vLLM servers found")
                return {}

        results = {}
        for server_type, config in self.server_configs.items():
            success = self.setup_forwarding(server_type, config)
            results[server_type] = success

        self.setup_complete = any(results.values())
        return results

    def get_active_servers(self) -> Dict[str, Dict]:
        """Get information about active vLLM servers"""
        active_servers = {}
        for server_type, config in self.server_configs.items():
            local_port = config.get('local_port', config['port'])
            node = config['node']

            # Try localhost first (for login node), then direct node access (for compute nodes)
            if self.test_port_connectivity(local_port):
                base_url = f'http://localhost:{local_port}/v1'
            elif self.test_node_connectivity(node, config['port']):
                base_url = f'http://{node}:{config["port"]}/v1'
                logger.info(f"Using direct node access: {base_url}")
            else:
                logger.warning(f"Cannot connect to {config['name']} on {node}:{config['port']}")
                continue

            active_servers[config['config_name']] = {
                'port': local_port,
                'base_url': base_url,
                'model': config['model'],
                'name': config['name'],
                'range_idx': config.get('range_idx', 0),
                'node': node
            }
        return active_servers


def quick_vllm_setup(vision_port: int = 8000, text_port: int = 8001) -> Tuple[bool, Dict[str, Dict]]:
    """
    Quick setup of vLLM cluster access with automatic fallback to direct node access.
    Uses global state to avoid repeated setup for the same port configuration.

    Args:
        vision_port: Port for vision model (default: 8000)
        text_port: Port for text model (default: 8001)

    Returns:
        Tuple of (success, active_servers_dict)
    """
    # Check if setup is already complete for these ports
    if is_vllm_setup_complete(vision_port, text_port):
        cached_servers = get_cached_vllm_servers(vision_port, text_port)
        logger.info(f"✅ vLLM already set up for ports {vision_port}/{text_port}, using cached configuration")
        return True, cached_servers

    logger.info(f"🔍 Setting up vLLM cluster access for ports {vision_port} (vision) and {text_port} (text)...")

    forwarder = get_vllm_forwarder()

    # Discover servers for specific ports
    if not forwarder.discover_servers(vision_port=vision_port, text_port=text_port):
        logger.error(f"No vLLM servers found on GPU nodes for ports {vision_port}, {text_port}")
        return False, {}

    logger.info(f"Found {len(forwarder.server_configs)} vLLM servers:")
    for server_type, config in forwarder.server_configs.items():
        logger.info(f"  - {config['name']} on {config['node']}:{config['port']}")

    # Check if we're on a compute node (SLURM_JOB_ID exists) or login node
    is_compute_node = 'SLURM_JOB_ID' in os.environ

    if is_compute_node:
        logger.info("🔧 Running on compute node - using direct node access (no port forwarding)")
        # Skip port forwarding, go directly to server discovery
        active_servers = forwarder.get_active_servers()
    else:
        logger.info("🔧 Running on login node - setting up port forwarding")
        # Set up forwarding as usual
        results = forwarder.setup_all_forwarding(vision_port=vision_port, text_port=text_port)
        success_count = sum(results.values())

        if success_count > 0:
            logger.info(f"✅ Successfully set up {success_count}/{len(results)} vLLM connections")
        else:
            logger.warning("⚠️ Port forwarding failed, trying direct node access...")

        # Get active servers (will try both localhost and direct node access)
        active_servers = forwarder.get_active_servers()

    if active_servers:
        logger.info(f"✅ Successfully connected to {len(active_servers)} vLLM servers:")
        # Log active endpoints
        for config_name, server_info in active_servers.items():
            logger.info(f"  - {server_info['name']}: {server_info['base_url']}")

        # Mark setup as complete and cache the results
        mark_vllm_setup_complete(vision_port, text_port, active_servers)
        return True, active_servers
    else:
        logger.error("❌ Failed to connect to any vLLM servers")
        return False, {}


def test_vllm_inference(server_configs: Dict[str, Dict]) -> bool:
    """
    Test inference on available vLLM servers.

    Args:
        server_configs: Dict from quick_vllm_setup()

    Returns:
        bool: True if at least one server is working
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("OpenAI package not available for inference testing")
        return False

    logger.info("🧪 Testing vLLM inference...")

    success_count = 0
    for config_name, server_info in server_configs.items():
        port = server_info['port']
        model_name = server_info['model']
        server_name = server_info['name']
        base_url = server_info['base_url']

        try:
            client = OpenAI(
                base_url=base_url,
                api_key="dummy",
            )

            # Quick test query
            test_query = "What is 2+2? Answer briefly."

            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
                    {"role": "user", "content": test_query}
                ],
                max_tokens=50,
                temperature=0.1
            )

            response = completion.choices[0].message.content
            logger.info(f"✅ {server_name}: {response.strip()}")
            success_count += 1

        except Exception as e:
            logger.warning(f"❌ {server_name} inference failed: {e}")

    total_servers = len(server_configs)
    logger.info(f"📊 Inference test results: {success_count}/{total_servers} servers working")

    return success_count > 0


def check_and_setup_vllm(vision_port: int = 8000, text_port: int = 8001) -> bool:
    """
    Set up vLLM cluster access for the specified ports.

    Args:
        vision_port: Port for vision model (default: 8000)
        text_port: Port for text model (default: 8001)

    Returns:
        bool: True if setup was successful or not needed, False if failed
    """
    try:
        print(f"🔍 Setting up vLLM cluster access for ports {vision_port} (vision) and {text_port} (text)...")

        try:
            # Quick setup using specified ports
            success, active_servers = quick_vllm_setup(vision_port=vision_port, text_port=text_port)

            if success:
                print("✅ vLLM cluster access established successfully")
                print(f"📋 Active servers: {len(active_servers)}")
                for _, server_info in active_servers.items():
                    print(f"  - {server_info['name']}: {server_info['port']}")

                # Quick inference test
                if test_vllm_inference(active_servers):
                    print("✅ vLLM inference test passed")
                else:
                    print("⚠️ vLLM inference test had issues, but proceeding...")

                return True
            else:
                print("❌ Failed to set up vLLM cluster access")
                print("⚠️ Proceeding anyway - may fall back to other APIs")
                return False

        except ImportError as e:
            print(f"⚠️ vLLM utilities not available: {e}")
            print("Proceeding without automatic vLLM setup")
            return False
        except Exception as e:
            print(f"⚠️ Error during vLLM setup: {e}")
            print("Proceeding anyway")
            return False

    except Exception as e:
        print(f"⚠️ Error in vLLM setup: {e}")
        return True  # Don't fail the whole process


def validate_vllm_ports(vision_port: int, text_port: int) -> Tuple[bool, str]:
    """
    Validate vLLM port configuration.

    Args:
        vision_port: Port for vision model
        text_port: Port for text model

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(vision_port, int) or not isinstance(text_port, int):
        return False, "Ports must be integers"

    if vision_port <= 0 or text_port <= 0:
        return False, "Ports must be positive integers"

    if vision_port == text_port:
        return False, "Vision and text ports must be different"

    if vision_port < 1024 or text_port < 1024:
        return False, "Ports should be >= 1024 to avoid system reserved ports"

    if vision_port > 65535 or text_port > 65535:
        return False, "Ports must be <= 65535"

    # Check if ports follow convention (even for vision, odd for text)
    if vision_port % 2 != 0:
        print(f"⚠️ Warning: Vision port {vision_port} is odd. Convention suggests even ports for vision models.")

    if text_port % 2 != 1:
        print(f"⚠️ Warning: Text port {text_port} is even. Convention suggests odd ports for text models.")

    return True, ""


def get_vllm_server_status(vision_port: int = 8000, text_port: int = 8001) -> Dict[str, Any]:
    """
    Get the status of vLLM servers without setting up connections.

    Args:
        vision_port: Port for vision model
        text_port: Port for text model

    Returns:
        Dictionary with server status information
    """
    try:
        forwarder = VLLMPortForwarding()

        # Discover servers
        if forwarder.discover_servers(vision_port=vision_port, text_port=text_port):
            status = {
                "servers_found": len(forwarder.server_configs),
                "servers": {}
            }

            for server_type, config in forwarder.server_configs.items():
                status["servers"][server_type] = {
                    "name": config["name"],
                    "node": config["node"],
                    "port": config["port"],
                    "model": config["model"]
                }

            return status
        else:
            return {"servers_found": 0, "servers": {}}

    except Exception as e:
        return {"error": str(e), "servers_found": 0, "servers": {}}


def setup_vllm_for_inference(
    vision_port: int = 8000,
    text_port: int = 8001,
    skip_setup: bool = False,
    verbose: bool = True
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Comprehensive vLLM setup for inference workflows.

    Args:
        vision_port: Port for vision model
        text_port: Port for text model
        skip_setup: Skip automatic vLLM setup
        verbose: Print status messages

    Returns:
        Tuple of (success, server_info_dict)
    """
    if verbose:
        print(f"🔧 vLLM Setup Configuration:")
        print(f"  Vision Port: {vision_port}")
        print(f"  Text Port: {text_port}")
        print(f"  Skip Setup: {skip_setup}")

    # Validate ports
    valid, error_msg = validate_vllm_ports(vision_port, text_port)
    if not valid:
        if verbose:
            print(f"❌ Port validation failed: {error_msg}")
        return False, None

    # Skip setup if requested
    if skip_setup:
        if verbose:
            print("⏭️ Skipping vLLM setup as requested")
        return True, None

    # Perform setup
    success = check_and_setup_vllm(vision_port=vision_port, text_port=text_port)

    if success:
        # Get server status
        server_info = get_vllm_server_status(vision_port=vision_port, text_port=text_port)
        return True, server_info
    else:
        return False, None


def print_vllm_setup_summary(
    vision_port: int,
    text_port: int,
    server_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Print a summary of the vLLM setup configuration.

    Args:
        vision_port: Vision model port
        text_port: Text model port
        server_info: Server information from setup
    """
    print("\n" + "="*50)
    print("🔧 vLLM SETUP SUMMARY")
    print("="*50)
    print(f"Vision Port: {vision_port}")
    print(f"Text Port: {text_port}")

    if server_info and "servers" in server_info:
        print(f"Servers Found: {server_info.get('servers_found', 0)}")

        for server_type, info in server_info["servers"].items():
            print(f"\n📡 {info['name']}:")
            print(f"  Node: {info['node']}")
            print(f"  Port: {info['port']}")
            print(f"  Model: {info['model']}")
    else:
        print("Server Info: Not available")

    print("="*50)


# Global instance for singleton pattern
_global_forwarder = None
_global_setup_state = {}  # Track setup status per port configuration


def get_vllm_forwarder() -> VLLMPortForwarding:
    """Get the global VLLM forwarder instance"""
    global _global_forwarder
    if _global_forwarder is None:
        _global_forwarder = VLLMPortForwarding()
    return _global_forwarder


def cleanup_vllm_forwarding():
    """Clean up global vLLM port forwarding"""
    global _global_forwarder, _global_setup_state
    if _global_forwarder is not None:
        _global_forwarder.cleanup()
        _global_forwarder = None
    _global_setup_state.clear()


def is_vllm_setup_complete(vision_port: int = 8000, text_port: int = 8001) -> bool:
    """Check if vLLM setup is already complete for given ports"""
    port_key = f"{vision_port}_{text_port}"
    return _global_setup_state.get(port_key, {}).get('setup_complete', False)


def mark_vllm_setup_complete(vision_port: int = 8000, text_port: int = 8001,
                           active_servers: Dict[str, Dict] = None):
    """Mark vLLM setup as complete for given ports"""
    port_key = f"{vision_port}_{text_port}"
    _global_setup_state[port_key] = {
        'setup_complete': True,
        'active_servers': active_servers or {},
        'setup_time': time.time()
    }


def get_cached_vllm_servers(vision_port: int = 8000, text_port: int = 8001) -> Optional[Dict[str, Dict]]:
    """Get cached vLLM server information if setup is complete"""
    port_key = f"{vision_port}_{text_port}"
    setup_info = _global_setup_state.get(port_key, {})
    if setup_info.get('setup_complete', False):
        return setup_info.get('active_servers', {})
    return None


def reset_vllm_setup(vision_port: int = 8000, text_port: int = 8001):
    """Reset vLLM setup state for given ports to force re-setup"""
    port_key = f"{vision_port}_{text_port}"
    if port_key in _global_setup_state:
        del _global_setup_state[port_key]
        logger.info(f"🔄 Reset vLLM setup state for ports {vision_port}/{text_port}")


def health_check_vllm_servers(vision_port: int = 8000, text_port: int = 8001) -> Dict[str, bool]:
    """
    Check health of cached vLLM servers.

    Returns:
        Dict mapping config_name to health status (True/False)
    """
    health_status = {}
    active_servers = get_cached_vllm_servers(vision_port, text_port)

    if not active_servers:
        return health_status

    for config_name, server_info in active_servers.items():
        base_url = server_info['base_url']
        try:
            health_url = base_url.replace('/v1', '/health')
            response = requests.get(health_url, timeout=5)
            health_status[config_name] = response.status_code == 200
        except Exception:
            health_status[config_name] = False

    return health_status


def reconnect_vllm_servers(vision_port: int = 8000, text_port: int = 8001,
                          force_reset: bool = False) -> Tuple[bool, Dict[str, Dict]]:
    """
    Attempt to reconnect to vLLM servers with automatic error recovery.

    Args:
        vision_port: Port for vision model
        text_port: Port for text model
        force_reset: If True, completely reset and re-setup connections

    Returns:
        Tuple of (success, active_servers_dict)
    """
    logger.info(f"🔄 Attempting vLLM reconnection for ports {vision_port}/{text_port}")

    if force_reset:
        logger.info("🔧 Force reset requested - clearing all cached state")
        reset_vllm_setup(vision_port, text_port)
        cleanup_vllm_forwarding()

        # Wait a moment for cleanup
        time.sleep(2)

        # Full setup from scratch
        return quick_vllm_setup(vision_port, text_port)

    # First, try health check of existing connections
    health_status = health_check_vllm_servers(vision_port, text_port)
    unhealthy_servers = [name for name, healthy in health_status.items() if not healthy]

    if not unhealthy_servers:
        logger.info("✅ All servers are healthy, no reconnection needed")
        return True, get_cached_vllm_servers(vision_port, text_port)

    logger.warning(f"⚠️ Unhealthy servers detected: {unhealthy_servers}")

    # Try to re-establish port forwarding for unhealthy servers
    forwarder = get_vllm_forwarder()

    # Clean up old forwarding processes
    forwarder.cleanup()

    # Wait for cleanup to complete
    time.sleep(3)

    # Re-setup forwarding
    logger.info("🔧 Re-establishing port forwarding...")
    success, new_servers = quick_vllm_setup(vision_port, text_port)

    if success:
        logger.info(f"✅ Reconnection successful - {len(new_servers)} servers active")
    else:
        logger.error("❌ Reconnection failed")

    return success, new_servers


def smart_vllm_request_with_reconnect(vision_port: int = 8000, text_port: int = 8001,
                                     max_reconnect_attempts: int = 2):
    """
    Decorator/context manager for vLLM requests that handles reconnection automatically.

    Args:
        vision_port: Port for vision model
        text_port: Port for text model
        max_reconnect_attempts: Maximum number of reconnection attempts

    Returns:
        Decorator function for API calls
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_reconnect_attempts + 1):
                try:
                    # Try the original function call
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()

                    # Check if this is a connection-related error
                    connection_errors = [
                        'connection', 'timeout', 'network', 'refused',
                        'unreachable', 'failed to connect', 'api connection'
                    ]

                    is_connection_error = any(err in error_str for err in connection_errors)

                    if is_connection_error and attempt < max_reconnect_attempts:
                        logger.warning(f"🔗 Connection error on attempt {attempt + 1}: {e}")
                        logger.info(f"🔄 Attempting reconnection ({attempt + 1}/{max_reconnect_attempts})")

                        # Try reconnection
                        force_reset = (attempt == max_reconnect_attempts - 1)  # Force reset on last attempt
                        success, _ = reconnect_vllm_servers(vision_port, text_port, force_reset=force_reset)

                        if success:
                            logger.info(f"✅ Reconnection attempt {attempt + 1} successful, retrying request")
                            # Wait a moment before retrying
                            await asyncio.sleep(2)
                            continue
                        else:
                            logger.warning(f"❌ Reconnection attempt {attempt + 1} failed")
                    else:
                        # Not a connection error, or we've exhausted retries
                        break

            # If we get here, all attempts failed
            logger.error(f"❌ Request failed after {max_reconnect_attempts + 1} attempts")
            raise last_exception

        return wrapper
    return decorator