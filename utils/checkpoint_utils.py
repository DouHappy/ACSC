"""
Checkpoint utilities for resuming interrupted tasks.
Provides functionality to save and load checkpoint states.
"""

import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

class CheckpointManager:
    """Manages checkpointing for long-running tasks."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(
        self,
        task_name: str,
        processed_index: int,
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_name: Optional[str] = None
    ) -> str:
        """
        Save checkpoint for a task.
        
        Args:
            task_name: Name of the task
            processed_indices: Set of processed data indices
            metadata: Additional metadata to save
            checkpoint_name: Custom checkpoint name, defaults to timestamp
            
        Returns:
            Path to saved checkpoint file
        """
        if checkpoint_name is None:
            checkpoint_name = f"{task_name}"
        
        checkpoint_data = {
            "task_name": task_name,
            "processed_index": processed_index,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint data from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data dictionary
        """
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        # Convert list back to set
        checkpoint_data['processed_indices'] = set(checkpoint_data['processed_indices'])
        
        return checkpoint_data
    
    def list_checkpoints(self, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Args:
            task_name: Filter by task name, None for all tasks
            
        Returns:
            List of checkpoint metadata
        """
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if task_name is None or data.get("task_name") == task_name:
                    checkpoints.append({
                        "path": str(checkpoint_file),
                        "task_name": data.get("task_name"),
                        "timestamp": data.get("timestamp"),
                        "total_processed": data.get("total_processed", 0)
                    })
            except Exception:
                continue
        
        # Sort by timestamp, most recent first
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints
    
    def get_latest_checkpoint(self, task_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest checkpoint for a specific task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Latest checkpoint data or None if no checkpoint exists
        """
        checkpoints = self.list_checkpoints(task_name)
        if not checkpoints:
            return None
        
        latest_path = checkpoints[0]["path"]
        return self.load_checkpoint(latest_path)
    
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Delete a checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            os.remove(checkpoint_path)
            return True
        except FileNotFoundError:
            return False
    
    def clean_old_checkpoints(self, task_name: str, keep_last: int = 3) -> int:
        """
        Clean old checkpoints, keeping only the most recent ones.
        
        Args:
            task_name: Name of the task
            keep_last: Number of recent checkpoints to keep
            
        Returns:
            Number of deleted checkpoints
        """
        checkpoints = self.list_checkpoints(task_name)
        if len(checkpoints) <= keep_last:
            return 0
        
        deleted_count = 0
        for checkpoint in checkpoints[keep_last:]:
            if self.delete_checkpoint(checkpoint["path"]):
                deleted_count += 1
        
        return deleted_count


class ResumeManager:
    """Manages resuming interrupted tasks."""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        """
        Initialize resume manager.
        
        Args:
            checkpoint_manager: CheckpointManager instance
        """
        self.checkpoint_manager = checkpoint_manager
        
    def get_resume_state(self, task_name: str) -> Dict[str, Any]:
        """
        Get resume state for a task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Dictionary with resume information
        """
        checkpoint = self.checkpoint_manager.get_latest_checkpoint(task_name)
        
        if checkpoint is None:
            return {
                "can_resume": False,
                "processed_indices": set(),
                "metadata": {},
                "total_processed": 0
            }
        
        return {
            "can_resume": True,
            "processed_indices": checkpoint["processed_indices"],
            "metadata": checkpoint["metadata"],
            "total_processed": checkpoint["total_processed"],
            "checkpoint_path": checkpoint.get("checkpoint_path", "")
        }
    
    def save_progress(
        self,
        task_name: str,
        processed_indices: Set[int],
        metadata: Optional[Dict[str, Any]] = None,
        auto_cleanup: bool = True
    ) -> str:
        """
        Save progress during task execution.
        
        Args:
            task_name: Name of the task
            processed_indices: Set of processed indices
            metadata: Additional metadata
            auto_cleanup: Whether to automatically clean old checkpoints
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            task_name=task_name,
            processed_indices=processed_indices,
            metadata=metadata
        )
        
        if auto_cleanup:
            self.checkpoint_manager.clean_old_checkpoints(task_name)
        
        return checkpoint_path


# Global instances for convenience
checkpoint_manager = CheckpointManager()
resume_manager = ResumeManager(checkpoint_manager)
