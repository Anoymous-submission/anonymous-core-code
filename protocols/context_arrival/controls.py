"""Independent intervention schedules; checkpoint identity is recorded separately."""


def resolve_domains(args):
    video, motion = {"None": (False, False), "Action": (False, True),
                     "Video": (True, False), "Full": (True, True)}[args.baseline_type]
    return (args.video_domain if args.video_domain is not None else (args.domain if video else "none"),
            args.motion_domain if args.motion_domain is not None else (args.domain if motion else "none"))


def visible_at(domain, block):
    return domain == "full" or (domain == "arrival" and block >= 1)
