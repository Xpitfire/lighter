from lighter.scheduler import Scheduler

if __name__ == '__main__':
    scheduler = Scheduler(path='runs/search/amused-frog',
                          experiment='lighter.experiment.DefaultExperiment')
    schedules = []
    for i, s in enumerate(scheduler):
        schedules.append(s)
        if i > 3:
            break
    assert schedules[0].config != schedules[1].config
    assert schedules[0].config != schedules[-1].config
    assert schedules[1].config != schedules[-1].config
