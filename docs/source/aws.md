*Created by:* Marta Karas ([martakarass](https://github.com/martakarass)) on 2022-02-18

*Credits*: Eli Jones ([biblicabeebli](https://github.com/biblicabeebli))

# Working with Forest on AWS

This page provides a quick start for using AWS EC2 and EBS to work  with Beiwe and other collaborators’ data. In particular, it includes guidelines to leverage AWS multicore computing environment and to run a Python file from multiple concurrent jobs running in the background. 

## Amazon EC2 and EBS

Amazon Web Services (AWS) is one of many cloud server providers. AWS offer various services,
including Amazon Elastic Compute Cloud ([Amazon EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html))
and Amazon Elastic Block Store ([Amazon EBS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AmazonEBS.html)).

-   EC2 provides virtual computing environments, known as instances,
    that come with preconfigured templates (including the operating
    system and additional software) and allow for various configurations
    of CPU, memory, storage, and networking capacity.
-   EBS provides storage volumes for use with EC2 instances. EBS is
    recommended for data that must be quickly accessible and requires
    long-term persistence (does not get deleted when you stop,
    hibernate, or terminate your instance).

## Starting setup

In this wiki page, we assume that:

-   EC2 instance with Linux (Amazon Linux 2 distribution) has been spun
    up,
-   ESB volume has been attached to EC2,
-   PEM private key to access EC2 has been shared with you,
-   user name and EC2 public DNS URL to be used have been shared with
    you; for example, the passage `<user name>@<EC2 public DNS URL>`
    could look in the lines of
    `ec2-user@ec2-1-1-1-1.compute-1.amazonaws.com`.

At the Onnela lab, it is likely all these four have been done by JP.

## SSH to an EC2 instance using a PEM key

To access EC2 instance through SSH, use a private key PEM file. Here,
the file is called `"DP-four-user.pem"`. It is likely the PEM file has
been shared with you via [Harvard Secure file
transfer](https://filetransfer.harvard.edu/).

Once you download the PEM file, you first need to modify permission
settings of that PEM file by typing the following in a terminal (note
the below line assumes that the PEM file is in the current working
directory):

``` sh
chmod 400 DP-four-user.pem
```

Then, you can access the instance by typing the following in a terminal
(note the below line assumes that the PEM file is in the current working
directory):

``` sh
ssh -i DP-four-user.pem ec2-user@ec2-1-1-1-1.compute-1.amazonaws.com
```

where `ec2-use` is a username and `ec2-1-1-1-1.compute-1.amazonaws.com`
is an EC2 public DNS URL. This particular DNS URL is made-up for security
reasons and we will use it throughout this wiki page.

Once you ssh, you can type `lscpu` to see CPU capacity and `lsmem` to
see memory of EC2.

## Make an EBS volume available for use on EC2 Linux instance

References used:
1.  [Make an Amazon EBS volume available for use on
    Linux](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html)
2.  [Manage user accounts on your Amazon Linux
    instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/managing-users.html)
3.  [Amazon EC2 key pairs and Linux
    instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html)
4.  [Understanding File
    Permissions](https://www.multacom.com/faq/password_protection/file_permissions.htm)

To make the newly attached EBS volume, we can format it to set up a file
system on it and then mount it, i.e. make it accessible from EC2 as if
it was a normal local disk. We follow steps from reference 1. Root user
privileges will be needed to perform these steps.

### Create a file system

Ssh to the EC2 instance as user with root privileges. Each Linux
instance launches with a default Linux system user account with root
privileges. For Amazon Linux 2, the user name is `ec2-user`.

``` sh
ssh -i DP-four-user.pem ec2-user@ec2-1-1-1-1.compute-1.amazonaws.com
```

Determine whether there is a file system on the volume. The FSTYPE
column shows the file system type for each device.

``` sh
sudo lsblk -f
```

    NAME          FSTYPE LABEL UUID                                 MOUNTPOINT
    nvme0n1                                                         
    ├─nvme0n1p1   xfs    /     1a1a1a1a-1a1a-1a1a-1a1a-1a1a1a1a1a1a /
    └─nvme0n1p128                                                   
    nvme1n1       

Here, `1a1a1a1a(...)` is a made-up security replacement for a passage of
alphanumeric characters. Here, there are two devices attached to the
instances – `nvme0n1` and `nvme1n1`. Device `nvme1n1` does not have a
file system. Partition `nvme0n1p1` on device `nvme0n1` is formatted
using the XFS file system.

We determine that `nvme1n1` is the newly attached EBS volume.

Create a file system on EBS volume.

``` sh
sudo mkfs -t xfs /dev/nvme1n1
```

### Mount the volume

Use the `mkdir` command to create a mount point directory for EBS
volume. The mount point is where the volume is located in the file
system tree and where you read and write files to after you mount the
volume. Here, the mount point directory is arbitrarily named
`dv1-mount-point`.

``` sh
sudo mkdir /dv1-mount-point
```

Mount EBS volume at the directory created in the previous step.

``` sh
sudo mount /dev/nvme1n1 /dv1-mount-point
```

After formatting and mounting, you should be able to see the following
output:

``` sh
sudo lsblk -f
```

    NAME          FSTYPE LABEL UUID                                 MOUNTPOINT
    nvme0n1                                                         
    ├─nvme0n1p1   xfs    /     1a1a1a1a-1a1a-1a1a-1a1a-1a1a1a1a1a1a /
    └─nvme0n1p128                                                   
    nvme1n1       xfs          2b2b2b2b-2b2b-2b2b-2b2b-2b2b2b2b2b2b /dv1-mount-point

Here, `2b2b2b2b(...)` is a made-up security replacement for a passage of
alphanumeric characters.

## Create new users at EC2

We assume that individuals who use EC2/EBS should generally access EC2
instance via one’s respective non-root user account and use root
account only when needed. Here, we show how to add a new user named
`marta`.

Assume we are ssh-ed to the EC2 instance as user with root privileges,
`ec2-user`. Create the user account and add it to the system. Then
switch to the new account.

``` sh
sudo adduser marta
sudo su - marta
```

Add the SSH public key to the user account. General guidance on
generating a key pair (public and private key) for use with AWS can
be found in reference 3. If you prefer for user `marta` to use the same
key pair as the `ec2-user`, you can retrieve public key from a
`ec2-user`’s private PEM key using the command below. The command places
the public key output in the clipboard (so you can paste it later to the
`authorized_keys` file).

``` sh
ssh-keygen -f DP-four-user.pem -y | pbcopy
```

First, create a directory in the user’s home directory for the SSH key
file. After the switch, the current directory should be the user’s home
directory (you can check it using `pwd` command). Change file
permissions for this newly created directory to `700` (only the owner
can read, write, or open the directory).

``` sh
mkdir .ssh 
chmod 700 .ssh
```

Create a file named `authorized_keys` in the `.ssh` directory and change
its file permissions to `600` (only the owner can read or write to the
file).

``` sh
touch .ssh/authorized_keys
chmod 600 .ssh/authorized_keys
```

Open the `authorized_keys` file and paste there a public key specific to
user `marta`. I personally use `vi` text editor and would open the file
using

``` sh
vi .ssh/authorized_keys
```

then type letter `i` to enter INSERT mode, then paste the public key,
then click `Escape` to exit INSERT mode, then type `:wq!` to save and
exit the file.

To switch back from `marta` to `ec2-user` user, use

``` sh
exit
```

Once an individual receives their private PEM key, they can modify
permission settings of the PEM file (see “SSH to an EC2 instance using a
PEM key” section) and ssh to the EC2 instance

``` sh
ssh -i <PEM key file name>.pem martar@ec2-1-1-1-1.compute-1.amazonaws.com
```

where `<PEM key file name>.pem` is a file name of the private PEM key for
user `marta`, possibly different than `DP-four-user.pem` of `ec2-user`.

## Review the file permissions of the new EBS volume mount

We review the file permissions of the new EBS volume mount to make sure
that all users (root and non-root) can write to the volume. For more
information about file permissions, see reference 4. or google for
similar.

Assume we are ssh-ed to the EC2 instance as user with root privileges,
`ec2-user`. This command that grants read-write-execute privileges to
all users on all EC2 instances that have the file system mounted.

``` sh
sudo chmod 777 /dv1-mount-point
```

You may want to create a subdirectory under `dv1-mount-point` for user
`marta`. The commands below create the subdirectory, make user `marta`
its owner, and grant privilege that only owner can write but everybody
else can read and execute files in the directory (recursively).

``` sh
sudo mkdir /dv1-mount-point/marta
sudo chown marta:marta /dv1-mount-point/marta
sudo chmod -R 755 /dv1-mount-point/marta
```

## Automatically mount an attached EBS volume after reboot

Following reference 1, we set up an automated mounting of the EBS volume
in case of a system reboot.

Assume we are ssh-ed to the EC2 instance as user with root privileges,
`ec2-user`. Create a backup of `/etc/fstab` file to be used in case it
is accidentally destroyed or deleted while being edited.

``` sh
sudo cp /etc/fstab /etc/fstab.orig
```

Use `lsblk` command to learn the UUID of the device that you want to
mount after reboot.

``` sh
sudo lsblk -f
```

    NAME          FSTYPE LABEL UUID                                 MOUNTPOINT
    nvme0n1                                                         
    ├─nvme0n1p1   xfs    /     1a1a1a1a-1a1a-1a1a-1a1a-1a1a1a1a1a1a /
    └─nvme0n1p128                                                   
    nvme1n1       xfs          2b2b2b2b-2b2b-2b2b-2b2b-2b2b2b2b2b2b /dv1-mount-point

Here, it is device `nvme1n1` and the UUID of interest is
`2b2b2b2b-2b2b-2b2b-2b2b-2b2b2b2b2b2b`.

Open the `/etc/fstab` using any text editor (I personally use `vi` text
editor; see above) and add the following entry, save and exit the file.

    UUID=2b2b2b2b-2b2b-2b2b-2b2b-2b2b2b2b2b2b  /data  xfs  defaults,nofail  0  2

See reference 1. for optional steps for checking whether the above
procedure worked out.

## Transfer files between local machine and EC2/EBS

There are several ways of transferring files between a local machine to
EC2/EBS remotes.

### SFTP (Secure File Transfer Protocol; SSH File Transfer Protocol)

On your local machine, go to the directory where your EC2 access private
PEM key is stored. The following command opens an SFTP connection to EC2
using user `marta` credentials.

``` sh
sftp -i DP-four-user.pem marta@ec2-1-1-1-1.compute-1.amazonaws.com
```

where `DP-four-user.pem` is the name of the private PEM key of user `marta`.

To get a list of all available SFTP commands, type `help`, or `?`. These
include:

-   `get` to download a file from the remote server to the local
    machine,
-   `put` to send a file from the local machine to the remote server.

The following command sends a file to `marta`’s user directory at EC2
instance:

``` sh
put /Users/martakaras/Desktop/dropbox.py /home/marta/
```

The following command sends a file to `marta`’s user directory at EBS:

``` sh
put /Users/martakaras/Desktop/dropbox.py /dv1-mount-point/marta/
```

### Cyberduck

[Cyberduck](https://cyberduck.io/) is an open-source application that
allows browsing and sending/downloading files to/from a remote server.

To use Cyberduck, download the Cyberduck application, open it and select
`Open Connection` button in top-left. A connection wizard window should
open. To connect with EC2,

-   specify `SFTP (SSH File Transfer Protocol)`,
-   set `Server` to EC2 public DNS URL (e.g.,
    `ec2-1-1-1-1.compute-1.amazonaws.com`),
-   set `Port` to `22`,
-   set `Username` of choice (e.g., `marta`),
-   set `Password` to the password corresponding to the username of
    choice (or keep the field empty if there is no password set for a
    particular EC2 user),
-   for `SSH Private Key`, navigate to and select the private PEM key file
    used to SSH to the EC2 instance (e.g., `DP-four-user.pem`),
-   select `Add to Keychain`,
-   click `Connect`.

If the connection is successful, a new Cyberduck window appears that allows
to browse the remote directory and transfer files from/to a local
machine via drag-and-drop.

### GitHub repo

A possible solution is to maintain a GitHub repository that is cloned in
both local machine and remote server. Note that even if the GitHub
repository is private, one should never push to the repository any
protected information.

### Headless Dropbox

A possible solution is to maintain a Dropbox directory that is synced
with a remote server. This approach is discussed in the next section.

## Transfer files between Dropbox and EC2/EBS

References:

1.  [Dropbox on Linux: installing from source, commands, and
    repositories](https://help.dropbox.com/installs-integrations/desktop/linux-commands)
2.  [How do I change the Dropbox directory on a headless GNU/Linux
    server?](https://superuser.com/questions/575550/how-do-i-change-the-dropbox-directory-on-a-headless-gnu-linux-server)

This section addresses a specific case scenario when a collaborator
shares with us data via Dropbox (e.g., from their Dropbox Business
account with MGB), the “target destination” of the data is the EBS
volume, but the data is so large that it cannot be first downloaded
locally and then uploaded to the EBS volume via the approaches listed
earlier. (For example, [Cyberduck does allow to transfer files from
Dropbox server to
EC2/EBS](https://www.ihaveapc.com/2020/01/how-to-use-cyberduck-to-connect-to-ftp-servers-dropbox-google-drive-and-amazon-s3/)
without an intermediate step of downloading to a local machine, but the
approach is not feasible for large files due to very long transferring
time and issues resulting from any internet connection interruptions.)

From my experience as of today, installing a “headless” Dropbox on the
EC2/EBS and syncing our Dropbox account which whom the data has been
shared – is an efficient way of file transferring in such a case.

### Note regarding Dropbox availability for use on EC2/EBS for all users

Many instructions to be found on the web provide a default way of
installing Dropbox via command line that installs the application in the
home directory of EC2 user that is conducting the installation (e.g., at
`/home/marta`). Even if the Dropbox daemon installer is downloaded and
launched in a directory shared among all users, the Dropbox daemon will
create Dropbox directory at, here, `/home/marta/` and sync the files
there. This is undesired if the files are to be accessible by all EC2
users (not only user `marta`).

The instructions below borrow from reference 2. and a few others to
force Dropbox daemon to create `Dropbox` directory and sync files at a
particular location on EBS.

### Note regarding syncing all files initially

Some comments on the web report that they were unable to sync
selectively before the syncing of all the files available at one’s
Dropbox account was initiated when the Dropbox daemon starts for the
first time. If the directory with collaborators data you want to
download to EC2/EBS is one of many directories you have within your
Dropbox account, it may mean some of the other files will, undesirably,
start getting synced.

A solution to avoid that would be to create a new Dropbox account that
only has the directory with collaborators shared with, and use that for
EC2/EBS syncing.

### Install and use Dropbox daemon

Download the Python script `dropbox.py` linked in the [reference 1.
website](https://help.dropbox.com/installs-integrations/desktop/linux-commands).
(Ctrl+F `python` on the website to locate the link). Upload the script
to a location that is accessible to all users. Here, I used

``` sh
put /Users/martakaras/Desktop/dropbox.py  /dv1-mount-point/shared/apps
```

where `/dv1-mount-point/shared/apps` is directory created earlier to
which all the users have full permissions set.

Set full permissions for all users to `dropbox.py` file.

``` sh
chmod 777 /dv1-mount-point/shared/apps/dropbox.py
```

Use `dropbox.py` to download, install, and start the Dropbox daemon. In
the below command, the `HOME=/dv1-mount-point/shared` part indicates
where the `Dropbox` directory will be located. Using this each time we
use `dropbox.py` is the key to having `Dropbox` directory syncing in a
location other than the default Dropbox daemon location (see “Note
regarding Dropbox availability for use on EC2/EBS for all users” above).

``` sh
HOME=/dv1-mount-point/shared /dv1-mount-point/shared/apps/dropbox.py start -i
```

where option `-i` do the auto-install of the Dropbox daemon if it is not
available on the system yet.

Follow the instructions from the console (including authorizing access
to your Dropbox account).

Once installation is completed, the files sync should start
automatically. For reference, syncing approx. 600 GB of data took me
approx. 2-3h. You can monitor the Dropbox daemon status, including the
syncing status, via `status` function.

``` sh
HOME=/dv1-mount-point/shared /dv1-mount-point/shared/apps/dropbox.py status
```

List of all functions available in `dropbox.py` to manipulate Dropbox
daemon can be found in reference 1.

Finally, the command below sets permissions to users for the Dropbox
directory, recursively.

``` sh
chmod -R 755 /dv1-mount-point/shared/Dropbox
```

## Install and use Anaconda on Linux instance

References:

1.  [Install Anaconda on Amazon
    Linux(EC2)](https://devopsmyway.com/install-anaconda-on-amazon-linuxec2/)
2. [how to specify new environment location for conda create](https://stackoverflow.com/questions/37926940/how-to-specify-new-environment-location-for-conda-create)

Anaconda is a distribution of the Python and R programming languages
that aims to simplify package management. Among others, Anaconda allows
to create multiple so-called environments, each of which may have its
own version of Python and its own versions of Python packages.

### Install Anaconda

Assume we are ssh-ed to the EC2 instance as user with root privileges,
`ec2-user`. Before installing Anaconda, make sure the following packages
are installed on the EB2 instance.

``` sh
sudo yum install libXcomposite libXcursor libXi libXtst libXrandr alsa-lib mesa-libEGL libXdamage mesa-libGL libXScrnSaver -y 
```

Navigate to a directory where every user has access to. Here, I used `/` directory. 

``` sh
cd /
```

Download the installer script for the latest version of Anaconda. Go to
<https://www.anaconda.com/products/individual>, find the one for Linux,
right-click on the download button and use “Copy Link Address” to copy
the download address link. Download the file using the link, e.g.

``` sh
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
```

Run the downloaded script.

-   When prompted for the installation location, use one that all
    users will have access to. I used `/opt/anaconda` (I later noted
    that internet suggests `/opt/anaconda/anaconda3` quite often instead).
-   When prompted
    `Do you wish the installer to initialize Anaconda3 by running conda init?`,
    select yes.

``` sh
sudo sh Anaconda3-2021.11-Linux-x86_64.sh
```

After installation completes, set the following permissions on the
directory where anaconda was installed.

``` sh
sudo chmod ugo+w /opt/anaconda
```

Add the following passage to user-specific `.bashrc` file located at
user’s home directory – for any user who should have access to that
installation of Anaconda.

-   To learn the home directory of a user, use `echo $HOME`. For
    example, `echo $HOME` prints `/home/ec2-user` if I am currently
    ssh-ed as `ec2-user` user, so I will paste the below passage into
    `/home/ec2-user/.bashrc`.
-   We may want to repeat this pasting procedure for every non-root user
    too (e.g., user `marta`).
-   Note the passage below may differ depending on the Anaconda
    installation location you chose. Assuming you were installing with
    the `ec2-user` user using `sudo`, your specific passage can be found
    in `/root/.bashrc`.

<!-- -->

    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/opt/anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/opt/anaconda/etc/profile.d/conda.sh" ]; then
            . "/opt/anaconda/etc/profile.d/conda.sh"
        else
            export PATH="/opt/anaconda/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<

To see an effect immediately (e.g., to be able to run `conda`
functions), source user’s `.bashrc` profile.

``` sh
source $HOME/.bashrc
```

From now on, you should be able to use `conda` functions, e.g.

``` sh
conda list
```

### Use Anaconda on Linux Instance

A neat thing about the setup that follows from the above installation is
that while Anaconda installation and `base` environment are shared among
all users, any environment a user create remains by default only visible
to that user.

Assume we are ssh-ed to the EC2 instance.

To create a new environment named `forest_main` with Python version
`3.8` use

``` sh
conda create --name forest_main python=3.8
```

To activate an environment named `forest_main` use

``` sh
conda activate forest_main
```

To deactivate an active environment use

``` sh
conda deactivate
```

To permanently delete an environment named `forest_main` use

``` sh
conda env remove -n forest_main
```

To look up environment locations use

``` sh
conda info -e
```

    # conda environments:
    #
    forest_main              /home/marta/.conda/envs/forest_main
    base                  *  /opt/anaconda

Here, two Anaconda environments are available: `base` (currently active,
as denoted by `*`) and `forest_main` (not currently active).

### Set directories for Anaconda environments and packages installation

Following the above installation setup, by default, Anaconda installs new packages in `/opt/anaconda/pkgs` and creates new environments within a within user-specific home directory (here, in `/home/marta/.conda/envs` for user `marta`). Both these directories are located in EC2 server files. As new packages get installed, the space on the EC2 (here, 8 GB) may run out. You can look up the default locations for Anaconda environments and packages by typing
```sh
conda info
```
```
# (...)
       package cache : /opt/anaconda/pkgs
                          /home/marta/.conda/pkgs
       envs directories : /home/marta/.conda/envs
                          /opt/anaconda/envs
# (...)
```
As a fix, for my user `marta` I configured Anaconda `pkgs_dirs` and `envs_dirs` variables to point to (previously created by me) specific directories on EBS volume: 
```sh
conda config --add pkgs_dirs '/dv1-mount-point/marta/apps/conda/pkgs'
conda config --add envs_dirs '/dv1-mount-point/marta/apps/conda/envs'
```
After the above, creating new environments and creating new users no longer takes space on EC2. I also see now: 
```sh
conda info
```
```
# (...)
          package cache : /dv1-mount-point/marta/apps/conda/pkgs
       envs directories : /dv1-mount-point/marta/apps/conda/envs
                          /home/marta/.conda/envs
                          /opt/anaconda/envs
# (...)
```
and an environment named `forest_main` created after above setup will yield: 
```sh
conda info -e
```
```
# conda environments:
#
forest_main              /dv1-mount-point/marta/apps/conda/envs/forest_main
base                  *  /opt/anaconda
```

## Install `forest` Python library

[Forest](https://github.com/onnela-lab/forest) is a Python library for
analyzing smartphone-based high-throughput digital phenotyping data
collected with the Beiwe platform. Forest implements methods as a Python
3.8 package. Forest is integrated into the Beiwe back-end on AWS but can
also be run locally.

Assume we are ssh-ed to the EC2 instance. Use the commands below to
activate Anaconda environment of choice (here, `forest_main` that has
Python `3.8` installed) and install `git`, `pip`.

``` sh
conda activate forest_main
conda install git pip
```

Install `forest` Python library. Use `@<branch name>` at the end to
specify the code branch of choice. Here, we install forest from `main`
branch.

``` sh
pip install git+https://github.com/onnela-lab/forest.git@main
```

## Run a Python file from multiple concurrent jobs running in the background

References:

1.  [How to run multiple background jobs in
    linux?](https://unix.stackexchange.com/questions/423805/how-to-run-multiple-background-jobs-in-linux)

### Running a Python file from a command line

Assume we are ssh-ed to the EC2 instance. The below commands activate
Anaconda environment of choice (here, `forest_main`), navigate to the
Python script location, run an exemplary Python file, and deactivate
the environment.

``` sh
cd /home/marta/_PROJECTS/beiwe_Nock_U01/py
conda activate forest_main
python download_data_gps_AWS.py 0
conda deactivate
```

where `0` is a value of an argument passed to `download_data_gps_AWS.py`
code that is read inside the Python file as

``` python
# (...)
job_id = int(sys.argv[1])
# (...)
```

### Wrap a Python file run into a bash script

The commands from the above subsection “Running a Python file from a command
line” can be wrapped into a following bash script, named
e.g. `run_download_data_gps_AWS.sh`:

``` sh
#!/bin/bash

# activate a specific conda environment
source /opt/anaconda/etc/profile.d/conda.sh
conda activate forest_main

# run script of interest
python download_data_gps_AWS.py $1

# deactivate a specific conda environment
conda deactivate
```

Assuming the above bash script is placed at the same directory as
`download_data_gps_AWS.py` file, we can use it to run
`download_data_gps_AWS.py` via:

``` sh
cd /home/marta/_PROJECTS/beiwe_Nock_U01/py
sh run_download_data_gps_AWS.sh 0
```

where `0` is a value of an argument passed to
`run_download_data_gps_AWS.sh` (and then to `download_data_gps_AWS.py`
via `$1`).

### Run multiple bash scripts concurrently in the background

The motivation for wrapping up a Python file run, together with Anaconda
environment activation/deactivation, into a bash script is that it will
allow us to run Python file multiple times
concurrently in the background with one command written in a compact way. To accomplish such task, consider the following command (note it is a long but essentially one line of code). 

``` sh
sh run_download_data_gps_AWS.sh 0 & sh run_download_data_gps_AWS.sh 1 & sh run_download_data_gps_AWS.sh 2 & sh run_download_data_gps_AWS.sh 3 & sh run_download_data_gps_AWS.sh 4 & sh run_download_data_gps_AWS.sh 5 & sh run_download_data_gps_AWS.sh 6 & sh run_download_data_gps_AWS.sh 7 & sh run_download_data_gps_AWS.sh 8 & sh run_download_data_gps_AWS.sh 9 & 
```
where `&` is used to start multiple background jobs. Then, in my workflow, I'd put the above line of code another bash script named e.g. `run_download_data_gps_MANY_AWS.sh`, and run it via

```sh
nohup sh run_download_data_gps_MANY_AWS.sh
```

where `nohup` ensures that the jobs keep running in the background even after we exit the terminal we used to run the above. 

The above example launches 10 concurrent background jobs, each with
a distinct value that is further used to define `job_id` variable value in
our `download_data_gps_AWS.py` Python script.

To see a potential use case, consider the case where a list of, say, N=100 Beiwe IDs is available.
The above workflow could be employed in a way that each job processes in
a loop a subset of 1/10 of N=100 Beiwe IDs using some Python-implemented
procedure. Here, a non-overlapping subset with 1/10 of N=100 Beiwe ID
could be defined inside `download_data_gps_AWS.py` based on `job_id`
variable value.
