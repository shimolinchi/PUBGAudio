import sys
from pathlib import Path
import copy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/synthesis'))
import generate_scenario_dataset as gen
import scenario_synthesis as motion


def track(cls=0,own=False,distance=150.):
    return dict(class_index=cls,source_role='self' if own else 'external',events=[],
        trajectory=dict(initial_xy=[distance,0],speed_knots=[[0,0],[30,0]],
            heading_knots_unwrapped_degrees=[[0,0],[30,0]],listener_bound=own),
        lifecycle=dict(startup_seconds=.3,move_seconds=3.8,final_stop_seconds=26.,shutdown_seconds=26.6,engine_end_seconds=29.))


def test_cabin_mask_retains_truth_and_coordinates():
    cfg=gen.configuration(); tracks=[track(),track(1,True)]
    paths=[motion.trajectory(t['trajectory'],30) for t in tracks]
    n=2997; powers=np.stack([np.full(n,1e-5),np.full(n,1e-2)])
    z=motion.frame_labels(tracks,paths,[],30,cfg,powers)
    assert z['track_activity'][0].all() and not z['track_observable'][0].any()
    assert z['track_distance_units'][0,0]==150
    assert not z['activity_loss_mask'][:,0,2].any()
    assert not z['track_direction_valid'][1].any()
    assert z['self_activity'][:,1].all()
    assert not z['activity'][:,1].any()


def test_weather_only_is_valid_negative_with_empty_tracks():
    z=motion.frame_labels([],[],[{'source_id':'weather'}],30,gen.configuration(),np.full((1,2997),.01))
    assert z['track_xy'].shape==(0,2997,2)
    assert not z['activity'].any() and not z['self_activity'].any()
    assert z['activity_loss_mask'][:,:3].all()
    assert not z['activity_loss_mask'][:,3].any()


def test_vehicle_curve_short_cruise_and_native_collision():
    t=track(1); t['trajectory']['speed_knots']=[[0,0],[10,20],[20,20],[30,0]]
    t['layers']={}
    rng=np.random.default_rng(1)
    motion.action_vehicle(t,30,rng,[dict(role='skid',id='native_skid')],[dict(id='native_crash')],1.)
    motion.curve_track(t,30,rng)
    path=motion.trajectory(t['trajectory'],30)
    assert np.all(path['distance']>=4)
    knots=t['trajectory']['speed_knots']
    cruise=sum(b-a for (a,v),(b,w) in zip(knots,knots[1:]) if v==w and v>0)
    assert cruise/(26.-3.8)<.03
    assert t['events'][-1]['role']=='collision'
    a,b=t['emergency_brake_seconds']
    assert b-a<=.501 and np.interp(b,path['time'],path['speed'])<.01
    assert t['layers']['skid']=='native_skid'
