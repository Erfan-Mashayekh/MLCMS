
from dataset import Dataset
from task import Task

if __name__ == '__main__':

    part = 4                # Choose between parts of task 2 : {1, 2, 3, 4(bonus)}

    nr_samples = 5000         # Number of samples
    nr_samples_plot = 1000     # Number of samples to plot
    sample = 10                # L+1 largest eigenfunctions
    radius = 5              # Radius around a point for distance computation

    print(f"\n Running part {part}.")
    dataset = Dataset(nr_samples, nr_samples_plot)
    task = Task(part, dataset, nr_samples, nr_samples_plot, sample, radius)

    if part == 1:
        task.part1()   # Periodic data set, this algorithm
    elif part == 2 or part == 4:
        task.part2()   # Swiss roll data set, (part 2) this algorithm, (part 4) Datafold algorithm
    elif part == 3:
        task.part3()   # Pedestrian trajectory data set, this algorithm
