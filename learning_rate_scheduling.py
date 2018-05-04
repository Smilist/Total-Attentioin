def learning_rate(epoch):

    # scheduling ver 5
    if epoch < 100:
        lr = 0.1
    elif epoch < 140:
        lr = 0.02
    elif epoch < 170:
        lr = 0.004
    else:
        lr = 0.0008

    '''
    # scheduling ver 8
    if epoch < 70:
        lr = 0.05
    elif epoch < 130:
        lr = 0.01
    elif epoch < 170:
        lr = 0.002
    else:
        lr = 0.0004
    '''

    return lr
