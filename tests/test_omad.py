import pytest
import numpy as np
from src.omad import OMADOrchestrator
from src.diffusion import DiffusionPolicy

def test_omad_orchestrator_initialization():
    agents = [DiffusionPolicy() for _ in range(2)]
    orchestrator = OMADOrchestrator(agents=agents, alpha=0.5)
    assert orchestrator.alpha == 0.5
    assert len(orchestrator.agents) == 2

def test_calculate_entropy_bonus():
    orchestrator = OMADOrchestrator(agents=[], alpha=1.0)
    
    # Same trajectories should have zero entropy bonus
    t1 = np.array([[1.0, 2.0]])
    t2 = np.array([[1.0, 2.0]])
    bonus_zero = orchestrator.calculate_entropy_bonus([t1, t2])
    assert bonus_zero == 0.0
    
    # Different trajectories should have positive bonus
    t3 = np.array([[2.0, 3.0]])
    bonus_pos = orchestrator.calculate_entropy_bonus([t1, t3])
    assert bonus_pos > 0.0

def test_joint_distributional_value():
    orchestrator = OMADOrchestrator(agents=[])
    t1 = np.random.randn(5, 2)
    t2 = np.random.randn(5, 2)
    
    val = orchestrator.joint_distributional_value([t1, t2], environment_state=None)
    assert isinstance(val, float)

def test_coordinate_logic():
    orchestrator = OMADOrchestrator(agents=[])
    t1 = np.array([[1.0, 1.0], [2.0, 2.0]])
    t2 = np.array([[3.0, 3.0], [4.0, 4.0]])
    
    coordinated = orchestrator.coordinate([t1, t2])
    
    # Should be the mean in current implementation
    expected = np.array([[2.0, 2.0], [3.0, 3.0]])
    np.testing.assert_array_almost_equal(coordinated, expected)

def test_diffusion_integration():
    policy = DiffusionPolicy(action_dim=2, horizon=4)
    trajectories = policy.get_trajectories("test context", num_samples=3)
    
    assert len(trajectories) == 3
    assert trajectories[0].shape == (4, 2)
    
    orchestrator = OMADOrchestrator(agents=[policy])
    consensus = orchestrator.coordinate(trajectories)
    assert consensus.shape == (4, 2)
