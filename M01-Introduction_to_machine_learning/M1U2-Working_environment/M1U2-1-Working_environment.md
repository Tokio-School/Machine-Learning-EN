# **Working environment**

M1U2 - Exercise 1

**What are we going to do?**

- Create a local VM instance (this is optional)
- Install JupyterLab and Python libraries in our working environment
- Troubleshooting

Remember to follow the instructions for the submission of assignments indicated in [Submission Instructions](https://github.com/Tokio-School/Machine-Learning-EN/blob/main/Submission_instructions.md).

**Instructions**

These instructions will show you how to install a virtual machine locally to complete the exercises in this course and to install JupyterLab and the necessary Python numerical libraries.

It is not necessary to create a local VM to perform the exercises, it is just another option. Other available options are:

- Google Colaboratory
- Creating a virtual environment in your Linux/Mac OS
- Using Windows Subsystem for Linux
- Using Google Cloud Platform's Vertex AI Workbench, or any other similar cloud service.
- Create a VM in the cloud or on a similar remote infrastructure.

We recommend using:

- Google Colaboratory
- If you use Windows: WSL with Ubuntu
- If you use Linux/Mac OS: Python Virtual Environment
- Create a local VM

**Create a local VM (optional)**

To create a local VM we have several options, depending on our operating system: Oracle VM VirtualBox, Vagrant, QEMU, Windows-specific environments Professional Edition and MacOs, etc.

If you are not familiar with using VMs, for simplicity's sake, we recommend that you use VirtualBox.

To do this, follow the installation instructions of the virtualisation software chosen for your operating system. E.g., for VirtualBox: [www.virtualbox.com](http://www.virtualbox.com/)

**Download Ubuntu**

As an operating system for the course, we will choose Ubuntu, mainly because of its simplicity. If you are sufficiently proficient in any other Linux distribution that supports the Python libraries to be used, you can use another distribution or even another OS, although we will not be able to support you in solving problems with the working environment.

Download Ubuntu Desktop for your architecture: [download Ubuntu Desktop](https://ubuntu.com/download/desktop)

**Create a VM**

Now follow the instructions of your virtualisation software to create a VM with that Ubuntu Desktop image. E.g., for VirtualBox: [Create your first VM](https://www.virtualbox.org/manual/UserManual.html#gui-createvm).

Create a VM with sufficient capacity, according to the resources available on your PC. A VM with at least 8 GB of memory and 20-30 GB of disk space is recommended. Depending on the virtualisation software, the VM resources can be changed afterwards with the VM switched off.

**Install Ubuntu on the VM**

Power on the VM for the first time and install the Ubuntu Desktop OS on it. Also, search online and install any additional guest components recommended for your virtualisation software on the VM.

These components usually allow us to perform tasks such as connecting USB devices to the VM, importing and exporting files from the VM, and generally supporting functions that make our work more comfortable.

E.g., for VirtualBox: [Host additions](https://www.virtualbox.org/manual/ch04.html)

**Python Environment**

For this course we use Python 3 exclusively. Ubuntu Desktop version 20+ uses Python 3 as its default version. Still, make sure you run the right Python version, use Pip for Python 3 and install Python 3 libraries.

_NOTE:_ In Google Colab or Vertex AI Platform you will already have an environment with the libraries installed.

**Virtual environment**

If we use a VM solely for this course, generally, we should not have problems with version conflicts, since we do not use it for other applications.

As it adds some extra complexity to the course, this step is optional. However, if you are sufficiently proficient with virtual environments for Python, you can use them. Some options are:

- Pipenv: [docs](https://pipenv-fork.readthedocs.io/en/latest/), [guide](https://realpython.com/pipenv-guide/)
- Venv: [docs](https://docs.python.org/3/library/venv.html), [guide](https://realpython.com/python-virtual-environments-a-primer/#using-virtual-environments) (note: for Python 3 use "venv", for Python 2 install "virtualenv")

**Updated list of dependencies**

Make sure that the local dependency listing is updated with the `sudo apt update` command before you start using the environment.

**Pip**

For a Python package manager, we will use Pip, which is already installed by default with Ubuntu. In some environments we can use the commands `python` and `pip` instead of `python3` and `pip3`.

If you have used an environment where your default Python version is Python 2, always make sure you use the right commands. In Ubuntu Desktop 20+, we should use `python3` and `pip3`.

The typical commands are:

- Install: `sudo apt-get install python3-pip`
- Check version: `pip3 --version`
- Update Pip (not usually necessary): `pip3 install --upgrade pip` (can be necessary to add `sudo -H` at the start)
- Install Python modules: `pip3 install name\_of\_the\_module`
- Update Python modules: `pip3 install --upgrade name\_of\_the\_module`
- Check installed modules and their versions: `pip3 freeze` or `pip3 freeze | grep name\_of\_the\_module' 

**Install the necessary libraries**

We will install NumPy, Matplotlib, Scikit-learn, and JupyterLab libraries:

`pip3 install numpy matplotlib scikit-learn jupyter jupyterlab`

If needed, we can find an installation guide with different options in each of the documentation pages of these programs.

**JupyterLab**

JupyterLab is an extension of Jupyter, which in turn is an evolution of IPython notebooks. All the documentation for JupyterLab is available in their [docs](https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html) online.

To start JupyterLab in each session, run the command `jupyter lab`

PS: Sometimes you will need to restart your VM or your VM session for it to recognise the new jupyter command for the first time.

Follow the instructions on your terminal to connect to the JupyterLab extension of the booted Jupyter server, which will be located at `http(s)://\<server:port\>/\<lab-location\>/lab`

At the end of the work session, simply shut down the Jupyter server by returning to the terminal with the keys `CTRL + C`.

To install JupyterLab extensions, follow the instructions in the documentation: [JupyterLab extensions](https://jupyterlab.readthedocs.io/en/stable/user/extensions.html)

**Potential problems**

In case of any problem, the best thing to do is to contact the teacher through a message on the platform. We will be more than happy to help you solve it as soon as possible.

**Deliverables**

Send a screenshot of your ready-to-use JupyterLab environment, in image format 😀.
