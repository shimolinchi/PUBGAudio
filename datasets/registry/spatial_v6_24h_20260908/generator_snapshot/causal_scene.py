"""V5 planner: random actor decisions followed by linked, spatially coherent events.

Preview simulation units are not calibrated game metres. No audio is rendered here.
"""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import motion_synthesis as motion
import footstep_acoustics as foot
from listener_view import make_view,bind_self_throw_direction
from event_acoustics import annotate_recipes

KINDS = ['footsteps', 'vehicle', 'gunfire', 'aircraft', 'action', 'explosion', 'fire', 'weather', 'warning']


def config(path):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    assert c['decision_step_seconds'] > 0 and c['trajectory_step_seconds'] > 0
    for row in c['spawn_probability_per_eligible_step'].values():
        if any(not 0 <= p <= 1 for p in row.values()):
            raise ValueError('Invalid probability')
    for name in ['density_weights', 'distance_mode_weights', 'curve_weights']:
        if min(c[name].values()) < 0 or sum(c[name].values()) <= 0:
            raise ValueError('Invalid weights: ' + name)
    return c


class Planner:
    def __init__(self, cfg, sources, seed):
        self.cfg, self.sources = cfg, sources
        self.rng = np.random.default_rng(seed)
        self.by_id = {s['id']: s for s in sources}
        self.roles, self.groups = defaultdict(list), defaultdict(list)
        for s in sources:
            self.roles[s['role']].append(s['id'])
            self.groups[s['source_group_id']].append(s)
        self.dt = cfg['trajectory_step_seconds']
        self.t = np.arange(round(cfg['clip_seconds'] / self.dt) + 1) * self.dt
        self.tracks, self.decisions, self.chains = [], [], []

    def uniform(self, bounds):
        return float(self.rng.uniform(*bounds))

    def choice(self, options, context='choice', time=None):
        if isinstance(options, dict):
            keys = list(options); w = np.array(list(options.values()), float); w /= w.sum()
        else:
            keys = list(options); w = np.ones(len(keys)) / len(keys)
        if not keys:
            raise ValueError('Empty candidate set: ' + context)
        selected = keys[int(self.rng.choice(len(keys), p=w))]
        self.decisions.append(dict(time=time, decision=context, candidates=keys, probabilities=w.tolist(), selected=selected))
        return selected

    def draw(self, p, context, time=None):
        selected = bool(self.rng.random() < p)
        self.decisions.append(dict(time=time, decision=context, probability=p, selected=selected))
        return selected

    def source(self, role, group=None):
        ids = self.roles[role] if group is None else [s['id'] for s in self.groups[group] if s['role'] == role]
        return self.choice(ids, 'source:' + role)

    def dwell(self, minimum, probability, maximum, context):
        d = minimum; step = self.cfg['decision_step_seconds']
        while d < maximum:
            d += step
            if self.draw(probability, 'switch:' + context):
                break
        return min(d, maximum)

    def base_track(self, kind, role='external', subtype='', group=''):
        x = dict(id=f'T{len(self.tracks)+1:03d}', kind=kind, class_index=KINDS.index(kind),
                 source_role=role, subtype=subtype, source_group_id=group, events=[], states=[],
                 gain_db=self.uniform(self.cfg['audio']['gain_jitter_db']))
        self.tracks.append(x)
        return x

    def event(self, track, role, start, duration=None, group=None, source_id=None, loop=False, **extra):
        sid = source_id or self.source(role, group)
        if duration is None:
            duration = self.by_id[sid]['duration_seconds']
        e = dict(role=role, source_id=sid, start=round(float(start), 6), duration=float(duration), loop=loop, **extra)
        track['events'].append(e)
        return e

    def point(self, role, radius=None):
        if role == 'self':
            return np.zeros(2)
        if radius is None:
            c = self.cfg['stationary_distance']
            far = self.draw(c['far_probability'], 'stationary_far')
            radius = self.uniform(c['far_units'] if far else c['near_units'])
        a = self.uniform([0, 2*np.pi])
        return radius * np.array([np.sin(a), np.cos(a)])

    def fixed(self, track, xy):
        self.set_path(track, np.tile(xy, (len(self.t), 1)), np.zeros(len(self.t)))

    def set_path(self, track, xy, speed, engine=None, brake=None):
        n = len(self.t)
        track['path'] = dict(time=self.t, xy=xy, speed=speed,
            engine_on=np.zeros(n) if engine is None else engine,
            brake=np.zeros(n) if brake is None else brake)

    def curved(self, track, speed, engine=None, brake=None):
        previous=track.get('curve_parameters')
        shape = previous['shape'] if previous else self.choice(self.cfg['curve_weights'], 'curve')
        mode = previous['mode'] if previous else self.choice(self.cfg['distance_mode_weights'], 'distance_mode')
        u = np.linspace(0, 1, 4001)
        if shape == 'straight':
            raw = np.column_stack((u*0, u))
        elif shape == 'arc':
            raw = np.column_stack((1-np.cos(u*1.4), np.sin(u*1.4)))
        elif shape == 'sine':
            raw = np.column_stack((.15*np.sin(u*2*np.pi), u))
        else:
            raw = np.column_stack((3*(1-u)**2*u*(-.35)+3*(1-u)*u*u*.35, u))
        distance = np.r_[0, np.cumsum((speed[:-1]+speed[1:]) * (self.dt/2))]
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
        raw *= max(distance[-1], 1e-6) / arc[-1]; arc *= max(distance[-1], 1e-6) / arc[-1]
        xy = np.column_stack([np.interp(distance, arc, raw[:,i]) for i in range(2)])
        if mode in ['far', 'near_pass', 'mid_pass']:
            bounds = {'far':[100,180], 'near_pass':[6,14], 'mid_pass':[25,45]}[mode]
            if track['kind']=='footsteps' and foot.settings(self.cfg):
                ac=foot.settings(self.cfg)
                radius=max((foot.radius_m(ac,e['role']) for e in track['events']),default=35)
                bounds=np.asarray(ac['closest_radius_fraction'][mode])*radius/ac['metres_per_scene_unit_assumed']
                track['distance_reference_radius_m_estimate']=radius
            closest = previous['closest'] if previous else self.uniform(bounds)
            xy[:,0] += closest - xy[:,0].min()
            xy[:,1] -= xy[-1,1] / 2
        elif mode == 'approach':
            xy -= xy[-1]; xy[:,0] += 10 - xy[:,0].min(); xy[:,1] -= 12
        else:
            xy[:,0] += 10 - xy[:,0].min(); xy[:,1] += 12
        angle = previous['angle'] if previous else self.uniform([0, 2*np.pi]); rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        xy = xy @ rot
        track.update(curve=shape, distance_mode=mode, listener_bound=track['source_role']=='self')
        track['curve_parameters']=dict(shape=shape,mode=mode,angle=angle,closest=locals().get('closest'))
        if track['listener_bound']:
            xy[:] = 0
        self.set_path(track, xy, speed, engine, brake)

    def duration(self, kind):
        buckets = self.cfg['duration_buckets_seconds'][kind]
        return self.uniform(buckets[self.choice(range(len(buckets)), 'duration_bucket:' + kind)])

    def footsteps(self, start, role='external', forced_curve=None):
        if self.cfg.get('body_actions'):
            from body_actions import footsteps
            return footsteps(self, start, role)
        c = self.cfg['footsteps']; end = min(start+self.duration('footsteps'), self.t[-1]-1)
        groups = [g for g in self.groups if g.startswith('footsteps:')]
        group = self.choice(groups, 'surface')
        available = {s['role'] for s in self.groups[group]}
        weights = {k:v for k,v in c['initial_weights'].items() if k in available}
        track = self.base_track('footsteps', role, 'locomotion', group); track['surface'] = group.split(':')[1]
        v = np.zeros(len(self.t)); current = start; last_state = self.choice(weights, 'gait')
        phase = self.uniform([0,1]); step_index = 0
        while current < end-.7:
            dur = self.dwell(c['minimum_dwell_seconds'], c['switch_probability_after_minimum'], c['maximum_dwell_seconds'], last_state)
            stop = min(end, current+dur); target = self.uniform(c['speed_units_per_second'][last_state])
            active = (self.t >= current) & (self.t < stop)
            ramp = min(c['speed_transition_seconds'], (stop-current)/3)
            v[active] = target * motion.smoothstep((self.t[active]-current)/ramp) * motion.smoothstep((stop-self.t[active])/ramp)
            track['states'].append(dict(start=current,end=stop,state=last_state))
            ids = [s['id'] for s in self.groups[group] if s['role']==last_state]
            for i in np.flatnonzero(active):
                cadence = min(3.65, 1.55*v[i]**.47) if v[i] >= .15 else 0
                phase += cadence * self.dt
                if phase >= 1:
                    phase -= 1
                    # Both alternating pools remain in one ground-material/gait family.
                    pool = ids[step_index % 2::2] or ids
                    self.event(track,last_state,self.t[i],source_id=self.choice(pool,'foot_variant'),
                               simulated_foot=['left','right'][step_index%2], foot_side_evidence='simulated_not_native')
                    step_index += 1
            pause = self.uniform(c['pause_duration_seconds']) if self.draw(c['pause_probability_after_bout'],'pause_after_bout') else .3
            track['states'].append(dict(start=stop,end=min(end,stop+pause),state='pause'))
            current = stop+pause
            # A pause separates any stance/gait change; there is no step-by-step alternation of walking/running.
            last_state = self.choice(weights, 'gait_after_pause')
        self.curved(track,v)
        track.update(start=start,end=end)
        return track

    def gun(self, start, role='external', position=None, attack_target=None):
        # Known fire modes only. Generic one-shot sources do not authorize automatic fire for bolt rifles.
        c = self.cfg['gunfire']; profiles = c['profiles']
        sr = 'local_shot' if role=='self' else 'shot'
        groups = [g for g in self.groups if g.startswith('gunfire:') and g.split(':')[1] in profiles and any(s['role']==sr for s in self.groups[g])]
        group = self.choice(groups, 'weapon'); weapon = group.split(':')[1]
        tr = self.base_track('gunfire',role,weapon,group)
        tr.update(weapon_asset_id=weapon, firing_interval_evidence='preview_profile_not_current_weapon_stat_reproduction')
        self.fixed(tr, self.point(role) if position is None else position)
        end = min(start+self.duration('gunfire'), self.t[-1]-3)
        if 'pattern_weights' in profiles[weapon]:
            from weapon_patterns import populate_fire
            return populate_fire(self, tr, profiles[weapon], sr, group, start, end, attack_target)
        cursor = start; ammunition = profiles[weapon]['magazine_limit']; shots_fired = 0
        while cursor < end and ammunition > 0:
            shots = self.choice(profiles[weapon]['burst_choices'], 'burst_shots')
            for _ in range(shots):
                if cursor >= end or ammunition == 0:
                    break
                self.event(tr,sr,cursor,group=group,attack_target=attack_target)
                cursor += profiles[weapon]['interval_seconds']; ammunition -= 1; shots_fired += 1
            cursor += self.uniform(c['rest_between_bursts_seconds'])
        tr.update(start=start,end=end,shots_fired=shots_fired,magazine_limit=profiles[weapon]['magazine_limit'])
        tr['states']=[dict(start=start,end=min(cursor,end),state='firing_bout')]
        if ammunition == 0 and cursor < end:
            tr['states'].append(dict(start=cursor,end=end,state='magazine_empty_no_reload_audio'))
        return tr

    def throwable(self,start,role='external',kind=None,radius=None):
        c=self.cfg['throwable']; kind=kind or self.choice(c['kind_weights'],'throwable_kind')
        xyz=self.point(role,radius); angle=self.uniform([0,2*np.pi]); length=self.uniform(c['throw_distance_units'])
        landing=xyz+length*np.array([np.sin(angle),np.cos(angle)])
        prep=self.base_track('action',role,kind+'_prepare'); self.fixed(prep,xyz)
        if kind=='molotov':
            self.event(prep,'molotov_ignite',start); release=start+2.; deadline=None
        else:
            self.event(prep,'pin',start,identity_from_audio='generic_pin_not_unique_throwable_type')
            cooked=self.draw(c['cook_probability'],'cooking',start)
            cooking=start+.25
            fuse=c['frag_fuse_seconds'] if kind=='frag' else c['flash_cooked_fuse_seconds'] if cooked else c['flash_uncooked_fuse_seconds']
            if cooked:
                self.event(prep,'cook',cooking)
                hold=self.uniform([.2,min(1.2,fuse-.7)])
                release=cooking+hold; deadline=cooking+fuse
            else:
                release=start+.6; deadline=release+fuse
        flight=self.uniform(c['flight_seconds']); impact=release+flight
        detonate=impact if kind=='molotov' else deadline if kind=='frag' else min(deadline,impact+c['flash_impact_delay_seconds'])
        # A cooked fuse may expire in flight; interpolate the actual position at that time.
        q=np.clip((self.t-release)/flight,0,1)
        xy=xyz[None,:]+q[:,None]*(landing-xyz)
        explosion_xy=xyz+np.clip((detonate-release)/flight,0,1)*(landing-xyz)
        xy[self.t>=detonate]=explosion_xy
        fx=self.base_track('fire' if kind=='molotov' else 'explosion','external',kind)
        self.set_path(fx,xy,np.where((self.t>release)&(self.t<min(impact,detonate)),length/flight,0))
        fx['source_role_evidence']='detached_effect_no_thrower_identity'
        if kind=='molotov':
            self.event(fx,'molotov_impact',impact)
            self.event(fx,'molotov_fire',impact+.05,duration=c['molotov_burn_seconds'],loop=True)
            fx['burn_duration_evidence']=c['molotov_duration_evidence']
        else:
            self.event(fx,kind+'_explosion',detonate)
        fx['states']=[dict(start=release,end=min(impact,detonate),state='flight'),dict(start=detonate,end=detonate+(c['molotov_burn_seconds'] if kind=='molotov' else 2),state='burn' if kind=='molotov' else 'detonated')]
        prep.update(start=start,end=release); fx.update(start=release,end=detonate+(c['molotov_burn_seconds'] if kind=='molotov' else 3))
        self.chains.append(dict(kind=kind,prepare=prep['id'],effect=fx['id'],pin_or_ignite=start,release=release,first_impact=impact,detonate=detonate,
                                fuse_deadline=deadline,release_xy=xyz,detonation_xy=explosion_xy))
        return prep

    def c4(self,start,role='external',radius=None):
        c=self.cfg['c4']; xy=self.point(role,radius)
        # A device is offset from the listener even when the direct placement action is self.
        device=xy+np.array([1.2,1.2]) if role=='self' else xy
        prep=self.base_track('action',role,'c4_deploy'); self.fixed(prep,xy)
        for r,offset in [('c4_switch_1',0),('c4_switch_2',1),('c4_switch_3',2),('c4_attach',3)]:
            self.event(prep,r,start+offset)
        armed=start+c['deploy_seconds']; explosion=armed+c['fuse_seconds']
        warning=self.base_track('warning','external','c4_beep'); self.fixed(warning,device)
        cursor=armed; knots=np.array(c['beep_interval_knots']); beeps=[]
        while cursor+self.by_id[self.roles['c4_beep'][0]]['duration_seconds'] < explosion:
            self.event(warning,'c4_beep',cursor); beeps.append(cursor)
            cursor+=float(np.interp(cursor-armed,knots[:,0],knots[:,1]))
        fx=self.base_track('explosion','external','c4'); self.fixed(fx,device); self.event(fx,'c4_explosion',explosion)
        prep.update(start=start,end=armed); warning.update(start=armed,end=explosion); fx.update(start=explosion,end=explosion+10)
        warning['states']=[dict(start=armed,end=explosion,state='armed_countdown')]
        self.chains.append(dict(kind='c4',prepare=prep['id'],warning=warning['id'],effect=fx['id'],activation=armed,detonate=explosion,beep_times=beeps,tempo_evidence=c['beep_interval_evidence']))
        return prep

    def aircraft(self):
        c=self.cfg['aircraft']; speed=self.uniform(c['speed_units_per_second']); offset=self.uniform(c['closest_offset_units'])
        xy=np.column_stack((np.full(len(self.t),offset),(self.t-self.t[-1]/2)*speed))
        a=self.uniform([0,2*np.pi]); xy=xy @ np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        tr=self.base_track('aircraft','environment','c130'); self.set_path(tr,xy,np.full(len(self.t),speed))
        self.event(tr,'aircraft_pass',0,duration=self.t[-1],loop=True)
        tr.update(start=0,end=self.t[-1],curve='straight',distance_mode='extremely_far_to_extremely_far')
        tr['states']=[dict(start=0,end=self.t[-1],state='constant_speed_flyby')]
        return tr

    def weather(self,start):
        tr=self.base_track('weather','environment','sandstorm'); self.fixed(tr,[0.,0.])
        tr.update(start=start,end=start+self.cfg['weather']['duration_seconds'])
        sid=self.choice([s['id'] for s in self.groups['weather:sandstorm']],'weather_variant')
        self.event(tr,'weather',start,duration=self.cfg['weather']['duration_seconds'],source_id=sid,loop=True)
        return tr

    def vehicle(self,start,role='external',force_coast=False,force_attack=None,force_collision=False):
        c=self.cfg['vehicle']; end=min(start+self.duration('vehicle'),self.t[-1]-3)
        suffix='_fpp' if role=='self' else ''
        groups=[g for g,ss in self.groups.items() if g.startswith('vehicle:') and {'idle','startup'+suffix,'shutdown'+suffix}.issubset({s['role'] for s in ss})]
        group=self.choice(groups,'vehicle_family'); tr=self.base_track('vehicle',role,group.split(':')[1],group)
        n=len(self.t); v=np.zeros(n); engine=np.zeros(n); brake=np.zeros(n)
        current=start; speed=0.; state='accelerate'; was_on=False; coast_done=False; first_acceleration=True
        startup_source=min([s for s in self.groups[group] if s['role']=='startup'+suffix],key=lambda s:s['duration_seconds'])
        startup_span=max(c.get('minimum_startup_seconds',3.5),startup_source['duration_seconds'])

        def segment(a,b,name,v0,v1,on,braking=False):
            mask=(self.t>=a)&(self.t<b)
            u=motion.smoothstep((self.t[mask]-a)/max(b-a,.01))
            v[mask]=v0+(v1-v0)*u; engine[mask]=on
            if braking: brake[mask]=1.
            tr['states'].append(dict(start=a,end=b,state=name))

        def start_engine(a,current_speed):
            span=startup_span
            self.event(tr,'startup'+suffix,a,source_id=startup_source['id'])
            next_speed=max(0,current_speed-.3*span)
            segment(a,a+span,'starting',current_speed,next_speed,True)
            return a+span,next_speed

        current,speed=start_engine(current,speed); was_on=True
        while current < end-5:
            if state=='accelerate' and not was_on:
                if current+startup_span+c['minimum_dwell_seconds']['accelerate']>end-4:
                    # Finish coasting/braking when there is no room for a complete restart and acceleration.
                    break
                current,speed=start_engine(current,speed); was_on=True
                if current >= end-5: break
            if state=='brake':
                duration=self.uniform(c['brake_seconds']); goal=0.
            elif state=='stopped':
                duration=self.uniform(c['stopped_seconds']); goal=0.
            else:
                maximum=c.get('maximum_dwell_seconds_by_state',{}).get(state,c['maximum_dwell_seconds'])
                duration=self.dwell(c['minimum_dwell_seconds'][state],c['switch_probability_after_minimum'][state],maximum,state)
                if state=='accelerate':
                    acceleration=self.uniform(c['accelerate_units_per_second_squared'])
                    if first_acceleration:
                        target_speed=self.uniform(c['cruise_speed_units_per_second'])
                        duration=min(c['maximum_dwell_seconds'],max(duration,(target_speed-speed)/acceleration))
                        tr['initial_motion_target_speed']=target_speed
                        first_acceleration=False
                    goal=min(c['maximum_speed_units_per_second'],speed+acceleration*min(duration,end-4-current))
                elif state=='decelerate': goal=max(0,speed-self.uniform(c['decelerate_units_per_second_squared'])*duration)
                elif state=='coast_off': goal=max(0,speed-self.uniform(c['coast_drag_units_per_second_squared'])*duration)
                else: goal=speed
            stop=min(current+duration,end-4)
            if state=='coast_off' and was_on:
                self.event(tr,'shutdown'+suffix,current,group=group); was_on=False; coast_done=True
            segment(current,stop,state,speed,goal,was_on,state=='brake')
            speed=goal; current=stop
            if speed < .2:
                state='accelerate' if state=='stopped' else 'stopped'
            elif force_coast and not coast_done:
                state='coast_off'
            else:
                legal={k:w for k,w in c['next_state_weights'].items() if k!=state and (was_on or k in ['coast_off','brake','accelerate'])}
                if c.get('acceleration_headroom_units_per_second') and speed>=c['maximum_speed_units_per_second']-c['acceleration_headroom_units_per_second']:
                    legal.pop('accelerate',None)
                state=self.choice(legal,'vehicle_next_state',current)
        if current < end-3:
            segment(current,end-3,'brake',speed,0,was_on,True)
        if was_on:
            self.event(tr,'shutdown'+suffix,end-3,group=group)
        tr['states'].append(dict(start=end-3,end=end,state='stopped_off'))
        layer_roles=['idle','roll','brake','skid']
        if c.get('available_layers_only',False):
            # A held-out source family may lack a tyre layer. Omit it explicitly;
            # never borrow a recording from another data partition.
            layer_roles=[r for r in layer_roles if r=='idle' or self.roles[r]]
            tr['available_vehicle_layers']=['engine']+[r for r in layer_roles if r!='idle']
        tr['layers']={r:self.source(r,group if r=='idle' else None) for r in layer_roles}
        tr['engine_phase_samples']=int(self.rng.integers(0,400000))
        moving=np.flatnonzero((v>4)&(self.t>start+5)&(self.t<end-7))
        contact_moving=np.flatnonzero((v>4)&(self.t>start+5)&(self.t<end-4.25))
        if len(contact_moving) and (force_collision or self.draw(c['collision_probability_per_vehicle_with_motion'],'vehicle_contact_episode')):
            off_moving=contact_moving[engine[contact_moving]==0]
            candidates=off_moving if force_coast and force_collision and len(off_moving) else contact_moving
            hit=float(self.t[candidates[len(candidates)//3]])
            self.event(tr,'collision',hit,cause='planned_obstacle_contact')
            dip=1-.65*motion.smoothstep((self.t-hit)/.35)
            recover=motion.smoothstep((self.t-hit-.5)/2)*float(np.interp(hit,self.t,engine)>.5)
            v*=np.where(self.t<hit,1,dip+.65*recover)
            brake[(self.t>=hit)&(self.t<hit+.35)]=1
            tr['states'].append(dict(start=hit,end=hit+.5,state='collision_contact'))
        self.curved(tr,v,engine,brake)
        if len(moving) and (force_attack or self.draw(c['attack_probability_per_vehicle_with_motion'],'vehicle_attack_episode')):
            hit=float(self.t[moving[len(moving)//2]])
            target=np.array([np.interp(hit,self.t,tr['path']['xy'][:,i]) for i in range(2)])
            attacker=self.gun(max(start+1,hit-2),'external',target+np.array([70.,50.]),tr['id'])
            # Restrict the attack to its causal bout before the hit; retain at least one real shot.
            attacker['events']=[e for e in attacker['events'] if e['start']<hit] or attacker['events'][:1]
            attacker['shots_fired']=len(attacker['events'])
            attacker['end']=max(e['start']+e['duration'] for e in attacker['events'])
            attacker['states']=[dict(start=attacker['start'],end=attacker['end'],state='firing_bout')]
            attack_end=max(e['start'] for e in attacker['events'])+.15
            hit=max(hit,attack_end)
            mode=force_attack
            if mode is None:
                mode='fatal' if self.draw(c['fatal_damage_probability_given_attack'],'fatal_given_attack') else 'tire' if self.draw(c['tire_failure_probability_given_attack'],'tire_given_attack') else 'body'
            chain=dict(kind='vehicle_attack',vehicle=tr['id'],shooter=attacker['id'],hit=hit,outcome=mode)
            if mode=='tire':
                self.event(tr,'tire_burst',hit,cause='incoming_gunfire',parent_track=attacker['id'])
                v*=1-(1-c['tire_speed_multiplier'])*motion.smoothstep((self.t-hit)/2)
                tr['states'].append(dict(start=hit,end=end,state='tire_damaged'))
            elif mode=='fatal':
                explode=hit+c['destroy_delay_seconds']; at_hit=float(np.interp(hit,self.t,v))
                mask=self.t>=hit; v[mask]=np.maximum(0,at_hit-.5*(self.t[mask]-hit))
                v*=1-motion.smoothstep((self.t-explode)/1.)
                engine[self.t>=hit]=0; brake[self.t>=hit]=0
                self.event(tr,'vehicle_fire',hit,duration=c['destroy_delay_seconds'],loop=True,cause='fatal_incoming_damage')
                self.event(tr,'vehicle_explosion',explode,cause='damage_countdown',parent_track=attacker['id'])
                # Never play a scripted restart/shutdown after a destroyed vehicle's engine has stopped.
                tr['events']=[e for e in tr['events'] if e['start']<hit or e['role'] in ['vehicle_fire','vehicle_explosion','collision','tire_burst']]
                tr['states']=[s for s in tr['states'] if s['start']<hit]
                for s in tr['states']: s['end']=min(s['end'],hit)
                tr['states'] += [dict(start=hit,end=explode,state='destroyed_burning'),dict(start=explode,end=self.t[-1],state='exploded')]
                chain.update(engine_failure=hit,detonate=explode,initial_health='already_damaged_preview_scenario')
            self.curved(tr,v,engine,brake)
            final_target=np.array([np.interp(hit,self.t,tr['path']['xy'][:,i]) for i in range(2)])
            self.fixed(attacker,final_target+np.array([70.,50.]))
            chain['hit_xy']=final_target
            if mode=='fatal':
                for effect_role,kind in [('vehicle_fire','fire'),('vehicle_explosion','explosion')]:
                    events=[e for e in tr['events'] if e['role']==effect_role]
                    if events:
                        fx=self.base_track(kind,'external',effect_role)
                        fx['path']=tr['path']; fx['events']=events
                        fx.update(start=events[0]['start'],end=events[0]['start']+events[0]['duration'],vehicle_parent=tr['id'])
                        chain[kind+'_track']=fx['id']
                tr['events']=[e for e in tr['events'] if e['role'] not in ['vehicle_fire','vehicle_explosion']]
            self.chains.append(chain)
        tr.update(start=start,end=max([end]+[e['start']+e['duration'] for e in tr['events']]))
        return tr

    def random_scene(self):
        density=self.choice(self.cfg['density_weights'],'density'); cfg=self.cfg
        horizons={'footsteps':35,'vehicle':44,'gunfire':16,'throwable':18,'c4':31,'aircraft':60,'weather':cfg['weather']['duration_seconds']}
        for tick in np.arange(0,cfg['clip_seconds'],cfg['decision_step_seconds']):
            busy=[t for t in self.tracks if t.get('start',0)<=tick<t.get('end',0)]
            for kind,p in cfg['spawn_probability_per_eligible_step'][density].items():
                reason=None
                if tick+horizons[kind]>cfg['clip_seconds']: reason='insufficient_complete_event_horizon'
                elif sum(t['kind'] in ['footsteps','vehicle','gunfire','action'] and not t.get('actor_parent') for t in busy)>=cfg['max_concurrent_actors'][density]: reason='actor_capacity'
                elif kind in ['aircraft','weather'] and any(t['kind']==kind for t in self.tracks): reason='one_per_clip'
                if reason:
                    self.decisions.append(dict(time=float(tick),decision='spawn:'+kind,eligible=False,reason=reason)); continue
                if not self.draw(p,'spawn:'+kind,float(tick)): continue
                role='external'
                own=[t for t in busy if t['source_role']=='self']
                if kind not in ['aircraft','weather'] and not own:
                    role='self' if self.draw(cfg['self_probability_when_eligible'],'direct_self',float(tick)) else 'external'
                cabin=any(t['kind']=='vehicle' and t['source_role']=='self' for t in busy)
                if cabin and kind=='footsteps' and not self.draw(cfg['masking']['cabin_foot_keep_probability'],'cabin_foot_admission',float(tick)):
                    continue
                begin=len(self.tracks); chain_begin=len(self.chains)
                if kind=='aircraft': self.aircraft()
                elif kind=='weather': self.weather(float(tick))
                else: getattr(self,{'footsteps':'footsteps','vehicle':'vehicle','gunfire':'gun','throwable':'throwable','c4':'c4'}[kind])(float(tick),role)
                if cabin and kind=='vehicle':
                    candidate=self.tracks[begin]
                    if np.linalg.norm(candidate['path']['xy'],axis=1).min()>=cfg['masking']['cabin_far_distance_units'] and not self.draw(cfg['masking']['cabin_far_vehicle_keep_probability'],'cabin_far_car_admission',float(tick)):
                        del self.tracks[begin:]; del self.chains[chain_begin:]; continue
                for tr in self.tracks[begin:]:
                    busy.append(tr)
        return density

    def result(self,identifier,density,selection='independent_random'):
        scene=dict(schema_version=5,id=identifier,seconds=self.cfg['clip_seconds'],density=density,selection=selection,
                    classes=KINDS,tracks=self.tracks,chains=self.chains,decisions=self.decisions)
        view=make_view(self)
        if view:scene['listener_view']=view
        if self.cfg['throwable'].get('self_throws_forward_at_release'):
            bind_self_throw_direction(scene)
        annotate_recipes(scene,self.cfg,self.by_id)
        return scene


def serializable(obj):
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,np.generic): return obj.item()
    raise TypeError(type(obj).__name__)
