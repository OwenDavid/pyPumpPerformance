# pyPumpPerformance
Interactive hydraulic pump performance PQ chart. Models the popular Hawe V60N series axial piston pump.

Default pump controller configuraiton with pressure compensator and torque limiter.

Pump characeteristics modeled from dataset combining pump dnyo-testing, manufacturers datasheet, designers test reports. 

Intended as an exploratory tool to aid hydraulic system design and comprehension. Utilised as a perforamnce optimistaion tool on grand prix racing yachts.

Thoughts/comments welcome.

## Future Improvements

* add gui wrapper around matplotlib figure
* refine matplolib updating of the figure - using blitting or similar animation technique
* check edge cases and limit impossible pump parameter combinations
* incoporate pump performance data from inline hydraulik test reports
