import sys
from collections import defaultdict
from cspProblem import CSP, Constraint
from searchGeneric import AStarSearcher
from cspConsistency import Search_with_AC_from_Cost_CSP

# generic domain scope
working_days = ['mon', 'tue', 'wed', 'thu', 'fri']
working_hours = ['9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm']

# global declaration
constraints = []
cost = defaultdict(int)
domains = {}
duration = {}


def before(task1, task2):  # return True if task1 is before task2 else False
    def before_(t1, t2):
        if t1//8 < t2//8:  # check day for t1 and t2
            return True
        # check time for t1 and t2
        elif t1//8 == t2//8 and (t1 % 8) + duration[task1] < (t2 % 8) + 1:
            return True
        return False
    return before_  # return before_ to before


def after(task1, task2):  # return True if task1 is after task2 else False
    def after_(t1, t2):
        if t1//8 > t2//8:  # check day for t1 and t2
            return True
        # check time for t1 and t2
        elif t1//8 == t2//8 and (t1 % 8) + 1 > (t2 % 8) + duration[task2]:
            return True
        return False
    return after_  # return after_ to after


def same_day(task1, task2):  # return True if task1 is on the same day as task2 else False
    def same_day_(t1, t2):
        # check for same day and sum of duration
        if t1//8 == t2//8 and duration[task1] + duration[task2] <= 8:
            return True
        return False
    return same_day_  # return same_day_ to same


def starts_at(task1, task2):  # return True if task1 starts just after task2 else False
    def starts_at_(t1, t2):
        # check for same day and t1-t2 must be duration of t2
        if t1//8 == t2//8 and (t1 % 8)-(t2 % 8) == duration[task2]:
            return True
        return False
    return starts_at_  # return starts_at_ to starts_at


def soft_constraint(task, t, c):  # creates a cost dictionary on runtime
    if c[0] == 'ends-by':
        cost_per_hour = int(c[-1])  # cost per hour form the input file
        # get the breakpoint number after which penalty applies
        d = working_days.index(c[1]) * len(working_hours) + \
            working_hours.index(c[2])
        extra_hours = 0 if t//8 < working_days.index(  # extra hours (next day(s))
            c[1]) else (t//8 - working_days.index(c[1])) * 16  # adding 16 hour with each passing day
        # create a cost dictionary with (taskname,starting tasktime) as keys and its cost as values
        cost[(task, t)] = (t + duration[task] - d + extra_hours) * \
            cost_per_hour if t + duration[task] >= d else 0
    return


def domain_duration(t, d):  # altering down the domain by taking duration into consideration
    for item in d.copy():
        # if the tasktime + its duration exceeds 5pm then remove the number for its domain
        if item % 8 + duration[t] > 8:
            d.remove(item)  # from dictionary
    return d  # return the updated/filtered domain set


def domain_num(c):
    dlen = len(working_days)
    tlen = len(working_hours)
    # check in workings_days
    if c in working_days:
        return [(working_days.index(c)*tlen+i) for i in range(tlen)]
    # check in working_hours
    elif c in working_hours:
        return [(i*tlen+working_hours.index(c)) for i in range(dlen)]
    # check for date-day range
    elif len(c.split()) == 4:
        c = c.split()[1:]
        from_day = c[0]
        from_time, to_day = c[1].split('-')
        to_time = c[-1]
        return [x for x in range(working_days.index(from_day) * tlen + working_hours.index(from_time),
                                 working_days.index(to_day) * tlen + working_hours.index(to_time))]
    # rest of all the cases
    else:
        c_list = c.split()
        con = c_list[0].split('-')[-1]
        # before
        if con == 'before':
            if len(c_list) == 2:
                return [(i*tlen+j) for j in range(working_hours.index(c_list[1])) for i in range(dlen)]
            else:
                return [x for x in range(working_days.index(c_list[1])*tlen+working_hours.index(c_list[2]) + 1)]
        # after
        elif con == 'after':
            if len(c_list) == 2:
                return [(i*tlen+j) for j in range(working_hours.index(c_list[1]), tlen) for i in range(dlen)]
            else:
                return [x for x in range(working_days.index(c_list[1])*tlen+working_hours.index(c_list[2]), dlen*tlen)]


# open file
fp = open(sys.argv[1], "r")
for line in fp:
    # ignore comments in the input file
    if '#' in line:
        line = line[:line.index('#')]
    line = list(map(lambda x: x.strip(), line.split(',')))
    # task and duration
    if line[0] == "task":
        task, dur = line[1].split()
        # creating domains dictionary
        domains[task] = set(range(len(working_days)*len(working_hours)))
        # creating duration dictionary
        duration[task] = int(dur)
    # binary constraint
    elif line[0] == "constraint":
        t1, c, t2 = line[1].split()
        c = c.replace('-', '_')
        b = c + "('" + t1 + "','" + t2 + "')"
        # creating constraints list
        constraints.append(Constraint(
            (t1.strip(), t2.strip()), eval(b)))
    # hard domain constraints
    elif line[0] == "domain":
        task = line[1].split()[0]
        # soft deadline constraints
        if line[1].split()[-1].isdigit():
            for t in domains[task]:
                soft_constraint(task, t, line[1].split()[1:])
        # hard constraints
        else:
            domains[task] = domains[task] & set(
                domain_num(' '.join(line[-1].split()[1:])))


fp.close()

for k in domains.keys():
    domains[k] = domains[k] & set(domain_duration(k, domains[k]))
# calling CSP class with domains, constraints and cost
csp = CSP(domains, constraints, cost)
# calling Search_with_AC_from_Cost_CSP class with csp
search_from_cost_csp = Search_with_AC_from_Cost_CSP(csp)
# calling search funstion of AStarSearcher class with search_from_cost_csp
path = AStarSearcher(search_from_cost_csp).search()

if path != None:
    final_cost = 0
    # sorting the path dictionary in increasing order of the tasknames
    path = {k: path.end().to_node[k] for k in sorted(path.end().to_node)}
    # printing out the output
    for v in path:
        print(v, ':', working_days[path[v]//8], ' ',
              working_hours[path[v] % 8], sep='')
        # calculating the final_cost
        final_cost += cost[(v, path[v])]
    print('cost:', final_cost, sep='')
else:
    print('No Solution')
