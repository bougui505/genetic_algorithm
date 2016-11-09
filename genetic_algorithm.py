#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2016-09-28 13:28:22 (UTC+0200)

import numpy
import scipy.optimize

class GA(object):
    """
    Genetic algorithm
    """
    def __init__(self, components, target, size=10, n_ensemble=1000,
                 crossing_freq=.8, mutation_freq=1., weights_experiment=None,
                 score_type=None):
        """
        • n_ensemble: number of ensembles to generate
        • size: size of the generated ensembles
        • weights_experiment: weight of each experiment. If
          weights_experiment=None, then all data in a are assumed to have a
          weight equal to one.
        • score_type: score type for each experiment: 'chi2' or 'qfactor'.
          If None, chi2 is chosen for all experiments.
        """
        self.crossing_freq = crossing_freq
        self.mutation_freq = mutation_freq
        self.components = components
        self.target = target
        self.size = size
        self.n_ensemble = n_ensemble
        if isinstance(self.components, tuple): # Multiple experimental data per
                                               # component
            self.n_components = self.components[0].shape[0]
            self.n_experiments = len(self.components)
            print "Number of experimental data type: %d"%self.n_experiments
        else:
            self.n_components = self.components.shape[0]
            self.n_experiments = 1
            self.components = (self.components, )
            self.target = (self.target, )
        self.ensemble = None
        self.model = None
        self.population = None
        self.score = None
        self.score_exp = [] # score for each experiment
        self.scale = None
        self.offset = None
        self.weights = None
        self.component_ids = None
        self.exp_id = None # Experimental index
        self.weights_experiment = weights_experiment
        if score_type is None:
            self.score_type = ('chi2', ) * self.n_experiments
        else:
            self.score_type = score_type

    def get_chi2(self, args):
        """
        Compute the Chi2
        """
        scale, offset = args
        chi2 = ((scale * (self.model + offset) -\
                self.target[self.exp_id][:, 1])**2 /\
                self.target[self.exp_id][:, 2]**2).mean()
        return chi2

    def get_qfactor(self, args):
        """
        Compute the Q-factor
        """
        scale, offset = args
        sumsq = lambda val: numpy.power(val, 2).sum()
        obs = self.target[self.exp_id][:, 1]
        calc = scale * (self.model + offset)
        return numpy.sqrt(sumsq(obs - calc) / sumsq(obs))

    def weight_ensemble(self, weights):
        """
        compute the weights that minimize the score (mixture of chi2 and
        Q-factor)
        """
        score_list = []
        for self.exp_id in range(self.n_experiments):
            ensemble = self.ensemble[self.exp_id]
            self.model = (ensemble * weights[:, None]).sum(axis=0)
            if self.score_type[self.exp_id] == 'chi2':
                chi2 = self.get_chi2((1., 0.))
                score_list.append(chi2)
            elif self.score_type[self.exp_id] == 'qfactor':
                qfactor = self.get_qfactor((1., 0.))
                score_list.append(qfactor)
        score = numpy.average(score_list, weights=self.weights_experiment)
        return score

    def get_models(self, exp_id):
        """
        return all the models for the given experiment id
        """
        if self.n_experiments > 1:
            population = self.population[:, exp_id, :]
            models = (population * numpy.asarray(self.weights)).sum(axis=1)
            models = numpy.asarray(self.scale)[:, exp_id] *\
                     (models + numpy.asarray(self.offset)[:, exp_id])
        else:
            models = (self.population[:, exp_id] * numpy.asarray(self.weights)[:,:,None]).sum(axis=1)
            models = numpy.asarray(self.scale) * (models + numpy.asarray(self.offset))
        return models

    def generate_ensemble(self):
        """
        generate a random ensemble
        """
        component_id = numpy.random.choice(self.n_components, self.size,
                                           replace=False)
        ensemble = []
        for exp_id in range(self.n_experiments):
            ensemble.append(self.components[exp_id][component_id])
        self.ensemble = tuple(ensemble)
        return component_id

    def generate_parents(self):
        """
        generate the initial population of parents
        """
        parents = []
        self.score = []
        self.scale = []
        self.offset = []
        self.weights = []
        self.component_ids = []
        for _ in range(self.n_ensemble):
            self.component_ids.append(self.generate_ensemble())
            parents.append(self.ensemble)
            score, scale, offset, weights = self.minimize_score()
            self.score.append(score)
            self.scale.append(scale)
            self.offset.append(offset)
            self.weights.append(weights)
        self.population = numpy.asarray(parents)

    def generate_offspring(self):
        """
        generate offspring from parents
        """
        offspring = numpy.copy(self.population)
        # Mutations
        mutant_index_list = numpy.random.choice(self.n_ensemble,
                                                size=int(self.n_ensemble *\
                                                    self.mutation_freq))
        offspring_component_ids = numpy.copy(self.component_ids)
        for mutant_index in mutant_index_list:
            gene_index = numpy.random.choice(self.size)
            new_gene_index = numpy.random.choice(self.n_components)
            for exp_id in range(self.n_experiments):
                offspring[mutant_index, exp_id, gene_index] =\
                                         self.components[exp_id][new_gene_index]
            offspring_component_ids[mutant_index][gene_index] = new_gene_index
        # Crossing over
        cross_index_list = numpy.random.choice(self.n_ensemble,
                                               size=int(self.n_ensemble *\
                                                    self.crossing_freq))
        for cross_index_1 in cross_index_list:
            cross_index_2 = numpy.random.choice(cross_index_list)
            gene_indices = numpy.random.choice(self.size,
                                               size=\
                                               numpy.random.choice(\
                                            range(1, self.size)), replace=False)
            part_1 = numpy.copy(offspring[cross_index_1, :, gene_indices])
            part_2 = numpy.copy(offspring[cross_index_2, :, gene_indices])
            ids_1 = numpy.copy(offspring_component_ids[cross_index_1]\
                                                                 [gene_indices])
            ids_2 = numpy.copy(offspring_component_ids[cross_index_2]\
                                                                 [gene_indices])
            offspring[cross_index_1, :, gene_indices] = part_2
            offspring[cross_index_2, :, gene_indices] = part_1
            offspring_component_ids[cross_index_1][gene_indices] = ids_2
            offspring_component_ids[cross_index_2][gene_indices] = ids_1
        self.population = numpy.concatenate((self.population, offspring))
        self.component_ids.extend(offspring_component_ids)
        self.component_ids = numpy.asarray(self.component_ids)
        for offspring_id, self.ensemble in enumerate(offspring):
            self.ensemble = tuple([numpy.asarray([e for e in self.ensemble[i]])\
                                  for i in range(self.n_experiments)])
            score, scale, offset, weights = self.minimize_score(weights=self.weights[offspring_id])
            self.score.append(score)
            self.scale.append(scale)
            self.offset.append(offset)
            self.weights.append(weights)
        self.score = numpy.asarray(self.score)
        self.score_exp = numpy.asarray(self.score_exp)
        self.scale = numpy.asarray(self.scale)
        self.offset = numpy.asarray(self.offset)
        self.weights = numpy.asarray(self.weights)

    def selection(self):
        """
        Select the self.n_ensemble best fitting individues
        """
        sorter = numpy.argsort(self.score)
        self.score = list(self.score[sorter][:self.n_ensemble])
        self.score_exp = [tuple(e) \
                         for e in self.score_exp[sorter][:self.n_ensemble]]
        self.scale = list(self.scale[sorter][:self.n_ensemble])
        self.offset = list(self.offset[sorter][:self.n_ensemble])
        self.weights = list(self.weights[sorter][:self.n_ensemble])
        self.population = self.population[sorter][:self.n_ensemble]
        self.component_ids = list(self.component_ids[sorter][:self.n_ensemble])

    def minimize_score(self, weights=None):
        """
        Minimize the score by finding the best weights then the best scaling and
        offset.
        """
        cons = ({'type': 'eq', 'fun': lambda x: sum(x) == 1})
        size = self.size
        if weights is None:
            weights = numpy.random.dirichlet(numpy.ones(size))
        res = scipy.optimize.minimize(self.weight_ensemble, weights,
                                      constraints=cons,
                                      bounds=[(0, 1), ] * size)
        weights = res.x
        score_list = []
        scales = []
        offsets = []
        for self.exp_id in range(self.n_experiments):
            self.model = (self.ensemble[self.exp_id] * weights[:, None]).\
                                                                     sum(axis=0)
            if self.score_type[self.exp_id] == 'chi2':
                res = scipy.optimize.minimize(self.get_chi2, [1., 0.])
                scale, offset = res.x
                scales.append(scale)
                offsets.append(offset)
                chi2 = self.get_chi2((scale, offset))
                score_list.append(chi2)
            elif self.score_type[self.exp_id] == 'qfactor':
                res = scipy.optimize.minimize(self.get_qfactor, [1., 0.])
                scale, offset = res.x
                scales.append(scale)
                offsets.append(offset)
                qfactor = self.get_qfactor((scale, offset))
                score_list.append(qfactor)
        self.score_exp.append(tuple(score_list))
        score = numpy.average(score_list, weights=self.weights_experiment)
        return score, scales, offsets, weights


if __name__ == '__main__':
    import ConfigParser
    import pickle
    import sys
    sys.path.append('.')
    import progress_reporting

    Config = ConfigParser.ConfigParser()
    Config.read(sys.argv[1])

    experiment_labels = list(set(Config.sections()) - set(('ga', )))
    experimental_weights = [float(Config.get(e, 'weight'))\
                            for e in experiment_labels]
    score_types = [Config.get(e, 'score_type')\
                   for e in experiment_labels]

    print "Experiments: %s"%experiment_labels
    print "weights: %s"%experimental_weights
    print "Score types: %s"%score_types

    def read_input(option):
        """
        Read the input files given in options 'target' and 'components' of the
        configuration file for each section.
        • option: 'target' or 'components'
        """
        output = {}
        for experiment_label in experiment_labels:
            output[experiment_label] =\
                        numpy.genfromtxt(Config.get(experiment_label, option))
        return output

    ### Read target:
    TARGETS = read_input('target')
    ### Read model components:
    COMPONENTS = read_input('components')

    def read_ga_parameters():
        """
        Read the parameters for the genetic algorithm from the configuration
        file
        """
        ga_params = {}
        for option in Config.options('ga'):
            ga_params[option] = Config.get('ga', option)
        return ga_params

    ga_params = read_ga_parameters()
    n_ensemble = int(ga_params['n_ensemble'])
    n_generation = int(ga_params['n_generation'])
    size = int(ga_params['size'])
    rsd_stop = float(ga_params['rsd']) # Stop criteria on Relative Standard Deviation
    window_size = int(ga_params['window_size']) # Size of the window to compute the RSD on

    def run_ga(experiment_list, n_generation, n_ensemble, size,
               weights_experiment, score_type):
        """
        Run the genetic algorithm for the given experiment list. The name of the
        experiments are the name given in the configuration file.
        """
        targets = tuple([TARGETS[k] for k in experiment_list])
        components = tuple([COMPONENTS[k] for k in experiment_list])
        ga = GA(components, targets, size=size, n_ensemble=n_ensemble,
                weights_experiment=weights_experiment, score_type=score_type)
        ga.generate_parents()
        score_list = []
        score_exp_list = [] # chi2 by experiment type
        progress = progress_reporting.Progress(n_generation, delta=1)
        for i in range(n_generation):
            ga.generate_offspring()
            ga.selection()
            score_min = ga.score[0]
            score_list.append(score_min)
            score_exp_list.append(ga.score_exp[0])
            if i >= window_size - 1:
                window = score_list[-window_size:]
                rsd = numpy.std(window) / numpy.mean(window)
                if rsd <= rsd_stop:
                    print "Stop criteria reached: rsd: %.4g <= %.4g"%(rsd, rsd_stop)
                    break
            else:
                rsd = numpy.nan
            progress.count(report="%s: %d: %s %s RSD: %.4g"%(experiment_list, i,
                                                   ga.score[0],
                                                   ga.score_exp[0], rsd))
        with open('ga_%s_%d_%d.dat'%('_'.join(experiment_list), n_ensemble, size), 'w') as outfile:
            pickle.dump(ga, outfile)
    run_ga(experiment_labels, n_generation, n_ensemble, size,
           experimental_weights, score_types)
