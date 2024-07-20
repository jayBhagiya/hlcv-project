# hlcv-project-GANs

## Getting Started

This file provides detailed instructions for submitting jobs on a high-performance computing (HPC) cluster using a remote SSH connection. The guide covers the necessary steps from establishing an SSH connection to setting up your development environment and managing job submissions and logs. For additional information, you can refer to these links:
- [HPC FAQ](https://wiki.cs.uni-saarland.de/en/HPC/faq)
- [Cluster Setup Guide](https://kingsx.cs.uni-saarland.de/index.php/s/ssmj33dxmgGsAYd)

## Connecting to the Cluster

**Note:** This guide assumes you are using a UNIX-based system. The instructions for connecting to the cluster from a Windows system will differ.

### Standard SSH Connection

To connect to the cluster, you need to be within the university's network. If you are at home, you can use the VPN.

Initiate a connection with the cluster by executing the following SSH command, substituting `<username>` with your specific username:

```sh
ssh <username>@conduit.cs.uni-saarland.de
```

Upon execution, enter your assigned password when prompted to complete the login process.

### Utilizing SSH Keys for Easier Access

#### Step 1: Generating an SSH Key
To avoid entering your password on each login, consider setting up SSH keys. Run these commands on your local machine, not the cluster:

```sh
ssh-keygen -t rsa -b 4096 -f ~/.ssh/hlcv_cluster
```

#### Step 2: Transferring the Public Key to the Cluster
Transfer the newly created public key to the cluster to enable key-based authentication:

```sh
ssh-copy-id -i ~/.ssh/hlcv_cluster.pub <username>@conduit.cs.uni-saarland.de
```

After completing this step, you should be able to log in using:

```sh
ssh <username>@conduit.cs.uni-saarland.de
```

without entering your password each time.

### Simplifying the SSH Command
To further simplify the SSH connection process, add an entry to your `~/.ssh/config` file:

```sh
Host hlcv_cluster
  HostName conduit.cs.uni-saarland.de
  User <username>
```

Replace `<username>` with your specific username. With this configuration, you can connect to the cluster by simply executing:

```sh
ssh hlcv_cluster
```

### Setting Up on the Cluster
Once connected to the cluster, clone the project repository:

```sh
git clone https://gitlab.cs.uni-saarland.de/jabh00002/hlcv-project-gans.git
```

You can make changes directly on the cluster using tools like VSCode with the [Remote-SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension or a terminal editor (vim or nano).

Now you can make changes on your machine, push to the repo, and pull the changes on the cluster.

### Installation of Miniconda
There is a folder named `env_setup` inside the project. Run the following commands:

#### Step 1: Install Miniconda
To initiate the setup, begin by installing Miniconda:

```sh
condor_submit setup.sub
```

This command installs Miniconda in the directory `~/miniconda3` and includes all the packages listed in `environment.yml` into an environment named `hlcv-ss24`. You can change the name in the `environment.yml` file. To add more packages, simply update the `environment.yml` file and re-execute the setup job.

For the first time, it will take time to install all the packages (around 50 mins). Once submitted, you can monitor your jobs [here](https://hpc-monitoring.cs.uni-saarland.de/). Log in with your same SSH credentials.

#### Step 2: Configuring the System Path (Optional)
To integrate conda into your system path, append the following line to your `~/.bashrc` file:

```sh
export PATH=$HOME/miniconda3/bin:$PATH
```

Or just run the following to activate the changes by sourcing the `~/.bashrc` file:

```sh
source ~/.bashrc
```

Sourcing the `~/.bashrc` file is important to let the system know that Miniconda has been installed. If, after sourcing, Miniconda doesn't work, exit the terminal and reconnect to the remote host.

#### Step 3: Submitting Scripts to the Cluster
With Miniconda configured, you can now submit scripts to the cluster. For example, to submit the Python scripts from the `src` folder, use the following command:

```sh
condor_submit trainer.sub
```

This command triggers `run_task.sh`, which selects the Conda environment specified in the `environment.yml` file and executes the Python script defined under `arguments` in `trainer.sub`.  To modify the Python script being executed, simply edit the value of `arguments` in the `trainer.sub` file. Here is what the `trainer.sub` file looks like:

```sh
universe                = docker
docker_image            = nvidia/cuda:11.4.3-runtime-ubuntu20.04
executable              = run_task.sh
arguments               = ~/hlcv-project-gans/src/trainer.py
environment             = PROJECT_ROOT=$ENV(PWD)
initialdir              = $ENV(PWD)
output                  = logs/trainer.$(ClusterId).$(ProcId).out
error                   = logs/trainer.$(ClusterId).$(ProcId).err
log                     = logs/trainer.$(ClusterId).log
transfer_output         = True
request_GPUs            = 2
request_CPUs            = 1
request_memory          = 4G
requirements            = UidDomain == "cs.uni-saarland.de"
getenv                  = HOME
+WantGPUHomeMounted     = true
queue 1
```
In a nutshell `trainer.sub` is desigend to run the `trainer.py` file (is in `src` folder) on the cluster. Remember if you want to run python script that is not in `src` then you need to change couple of things. 

- First change the file name under `arguments` in `trainer.sub`.
- Below is the last line of `run_task.sh`. Here `${SCRIPT_NAME}` is the value that we put in `arguments` in `trainer.sub`. So I would suggest to use the abosolute path as seen in above example.

  ```sh
  python ${SCRIPT_NAME}
  ```

You can change the value of `request_GPUs`, `request_CPUs` or `request_memory` as per your requirement of GPU, CPU or memory in the `trainer.sub` file.

### Monitoring Job Execution and Logs
#### Accessing Job Logs
Upon successful job execution, log files are stored within the `logs/` directory. To review the output of a specific job, use the following command, replacing `<job_id>` with the actual job ID:

```sh
less logs/run.<job_id>.0.out
```

#### Real-time Monitoring of Ongoing Processes
For real-time monitoring of a process currently in execution, use the following command, again substituting `<job_id>` with the correct job ID:

```sh
tail -f logs/run.<job_id>.0.out
```

#### Other Useful Commands
Check the state of your job in the Condor queue:

```sh
condor_q
```

Analyze how many machines can run your job or if there are problems:

```sh
condor_q -analyze
```

```sh
condor_q -better
```

Overview of machines in the cluster:

```sh
condor_status
```

To terminate a job:

Find the job ID:

```sh
condor_q –nobatch
```

Terminate the job:

```sh
condor_rm <job_id>
```

### How to View Files on the Cluster
Since the cluster does not have a graphical user interface (GUI), alternative methods are needed to view files such as plots. You can either transfer these files to your local computer or use an application that provides a GUI through SSH.

#### For Linux Users
If you're using Linux, your file explorer likely supports SFTP. Connect to `sftp://conduit.cs.uni-saarland.de` to access and view all files on the cluster directly from your file explorer.

#### For Windows Users
On Windows, you can use a file transfer application with SFTP support, such as FileZilla or WinSCP. These programs offer a graphical interface for transferring files from the cluster to your computer.

#### Using Visual Studio Code
Regardless of your operating system, you can use Visual Studio Code with the Remote-SSH extension. This setup allows you to develop directly on the cluster and view files, including images, as if you were working locally.

#### Copying Files Using scp
Another method is to copy files to your local machine using the scp command. This is useful for viewing files on your own computer. To copy a file from the cluster to your machine, use the following command. Replace `<username>` with your actual username, `/path/to/your/file/on/remote` with the file's path on the cluster, and `/copy/to/here/` with the destination path on your machine:

```sh
scp <username>@conduit.cs.uni-saarland.de:/path/to/your/file/on/remote /copy/to/here/
```

If you have previously set up an entry in your `~/.ssh/config` file, the command simplifies to:

```sh
scp hlcv_cluster:/path/to/your/file/on/remote /copy/to/here/
```

This command uses the alias `hlcv_cluster` that you defined in your SSH configuration.

#### Using Git
If you have set up your own repository, you can also add the files to your repository and manage them through Git.
