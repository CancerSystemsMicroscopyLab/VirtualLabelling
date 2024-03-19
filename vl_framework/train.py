import time
from models import create_model
import os


def print_log(logger, message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')


def train(opt, dataset):
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    logger = open(os.path.join(save_dir, 'log.txt'), 'w+')
    print_log(logger, opt.name)
    logger.close()

    model = create_model(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        # Training step
        opt.phase = 'train'
        for i, data in enumerate(dataset):
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

        print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
        model.save('latest')

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
