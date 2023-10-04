# cleaning-AI
An AI implemention of a genetic algorithm to optimise the fitness of a population of vacuuming agents, a.k.a. cleaners, tasked with cleaning an area. The rules of the world are as 
follows: a cleaner can move forwards or backwards in discrete steps; it can rotate in either direction by 90 degrees; when driving over dirty area it automatically picks up the dirt 
(unless its bin is full); cleaners run on a battery that needs to be recharged periodically at a charging station, where bin also is emptied. The objective is to clean maximal area. 
There is another population of cleaners on the field, competing for cleaning credits. 

The AI finds behaviours which lead to the most cleanig done. Each cleaner acts as an independent agent and its behaviour is learned through the process of simulated fitness selection and evolution


To play the cleaning game run the cleaners.py code. Make sure in the settings.py file that at least one of the agents is set to my_agent.py. 
