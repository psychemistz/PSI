#!/usr/bin/env python
from optparse import OptionParser
from numpy import *
import os

import lwa
import sr
import rge

def read_expression_file(expression_file):
    "Reads EXPRESSION_FILE, which is a tab-delimited file containing gene expression values. The first line is ignored (assumed to be a list of the experiment names). The first field in every line of the remaining lines is the name of the probe, and the rest are real numbers representing expression values in log-space."
    try:
        f = open(expression_file, "rb")
        first_line = f.readline()
        probes = []
        expr = []
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) > 0:
                probe_id = fields[0]
                expr_values = map(float, fields[1:])
                probes.append(probe_id)
                expr.append(expr_values)
        f.close()
        expr = array(expr)
        return expr, probes, first_line
    except:
        print("Error reading %s!"%expression_file)
        return None

[MODE_IMPUTE, MODE_SELECT, MODE_EXTRACT] = range(3)

def main():
    usage = "usage: %prog [options] input_file output_file"
    parser = OptionParser(usage)
    parser.add_option("-s", "--select", dest="probes", default="5",
                      help="select the specified number of probes and learn an imputation model")
    parser.add_option("-i", "--impute", dest="imputation_model",
                      help="impute using the model saved in IMPUTATION_MODEL")
    parser.add_option("-o", "--output", dest="model_output",
                      help="name of file to save the imputation model")
    parser.add_option("-m", "--method", dest="method", default="SR",
                      help="use METHOD (LWA, RGE or SR) for PSI")
    parser.add_option("-f", "--force", dest="force_select",
                      help="a file of probes that must be selected")
    parser.add_option("-e", "--extract", dest="probes_filename",
                      help="extract the probes in the given file from the input, and saved the subset in the output")
    parser.add_option("-g", "--gtpcc", dest="ground_truth",
                      help="compute the median Pearson correlation coefficient between the imputations and the ground truth, not including the selected probes")
    (options, args) = parser.parse_args()
    if len(args) < 2:
        parser.print_usage()
        return
    expression_file = args[0]
    output_file = args[1]
    if options.imputation_model is not None:
        mode = MODE_IMPUTE
        imputation_model_filename = options.imputation_model
        gt_filename = options.ground_truth
    elif options.probes_filename is not None:
        mode = MODE_EXTRACT
        probes_filename = options.probes_filename
    else:
        mode = MODE_SELECT
        method = options.method.upper()
        if method not in ['LWA','RGE','SR']:
            parser.error("Method must be one of: LWA, RGE, SR")
        psim = psi_initialize(method)
        model_output = options.model_output            
        num_probes = options.probes
        try:
            num_probes = int(num_probes)
        except:
            parser.error("Number of probes must be an integer.")
        print("Selecting %d probes using %s."%(num_probes, method))
        print("Probe list will be saved in %s."%output_file)
        if model_output is not None:
            print("Imputation model will be saved in %s."%model_output)
        force_include_filename = options.force_select
    print("Reading input...")
    read_result = read_expression_file(expression_file)
    if read_result is None:
        return
    expr, probes, exps = read_result
    print("Input read successfully with %d probes and %d experiments."%(expr.shape[0], expr.shape[1]))
    if expr is None:
        return
    if mode == MODE_SELECT:
        force_inds = []
        if force_include_filename is not None:
            force_probes = read_probes_from_file(force_include_filename)
            if force_probes is not None:
                inds = find_probes(probes, force_probes)
                if inds is not None:
                    print("Forcing inclusion of %d probes."%len(inds))
                    force_inds = inds
        probe_list, imp_model = psi_select(expr, probes, num_probes, psim, force_include = force_inds)
        if save_probe_list(probe_list, output_file):
            print("Probe list saved in %s."%output_file)
        if model_output is not None:
            if save_imputation_model(method, imp_model, num_probes, probes, model_output):
                print("Imputation model saved in %s."%model_output)
    elif mode == MODE_IMPUTE:
        method, output_probes, input_probes, imputation_model = read_imputation_model(imputation_model_filename)
        if imputation_model is None:
            return
        expr2 = extract_probes(expr, probes, input_probes)
        if expr2 is None:
            return
        psim = psi_initialize(method)
        imputation = psim.impute_from_params(imputation_model, expr2)
        if save_imputation_output(output_probes, imputation, exps, output_file):
            print("Imputation saved in %s."%output_file)
        if gt_filename is not None:
            read_result = read_expression_file(gt_filename)
            if read_result is None:
                return
            gtexpr, gtprobes, gtexps = read_result
            if (gtprobes == output_probes):
                if (gtexps == exps):
                    gtpcc = compute_gtpcc(gtexpr, imputation, output_probes, input_probes)
                    print("gtPCC: %.4f"%gtpcc)
                else:
                    print("Error comparing imputation to ground truth: experiment names not the same.")
            else:
                print("Error comparing imputation to ground truth: probe names not the same.")
    elif mode == MODE_EXTRACT:
        probes2 = read_probes_from_file(probes_filename)
        expr2 = extract_probes(expr, probes, probes2)
        if save_imputation_output(probes2, expr2, exps, output_file):
            print("Extracted expression saved in %s."%output_file)

def compute_gtpcc(expr1, expr2, output_probes, input_probes):
    exclude_set = set(input_probes)
    pcc_inds = [i for i in xrange(len(output_probes)) if output_probes[i] not in exclude_set]
    expr1_0 = expr1[pcc_inds,:] - mean(expr1[pcc_inds,:],1)[:,newaxis]
    expr2_0 = expr2[pcc_inds,:] - mean(expr2[pcc_inds,:],1)[:,newaxis]
    pccs = sum(expr1_0*expr2_0,1)/sqrt(sum(expr1_0**2,1)*sum(expr2_0**2,1))
    gtpcc = median(pccs)
    return gtpcc

def find_probes(original_probes, probes):
    d = dict(zip(original_probes,range(len(original_probes))))
    inds = []
    for probe in probes:
        if d.has_key(probe):
            inds.append(d[probe])
        else:
            print("Error! Probe %s not found."%probe)
            return
    return inds

def extract_probes(expr, original_probes, probes):
    inds = find_probes(original_probes, probes)
    if inds is not None:
        return expr[inds,:]

def save_imputation_output(probes, imputation, first_line, filename):
    try:
        f = open(filename, "wb")
        f.write(first_line)
        for i in xrange(len(probes)):
            f.write('\t'.join([probes[i]]+['%s'%x for x in imputation[i]])+os.linesep)
        f.close()
        return True
    except:
        print("Error saving imputation in %s."%filename)

def read_probes_from_file(filename):
    try:
        f = open(filename, "rb")
        probes = []
        for line in f:
            probes.append(line.strip())
        f.close()
        return probes
    except:
        print("Error reading probes from file %s."%filename)

def read_imputation_model(filename):
    try:
        f = open(filename, "rb")
        method = f.readline().strip()
        if method in ['RGE','SR']:
            input_probes = f.readline().strip().split("\t")[1:-1]
            probes = []
            rows = []
            for line in f:
                line2 = line.strip().split('\t')
                if len(line2) > 0:
                    probes.append(line2[0])
                    rows.append(map(float, line2[1:]))
            f.close()
            params = array(rows)
        elif method in ['LWA']:
            input_probes = f.readline().strip().split("\t")
            kw = float(f.readline().strip())
            probes = []
            rows = []
            for line in f:
                line2 = line.strip().split('\t')
                if len(line2) > 0:
                    probes.append(line2[0])
                    rows.append(map(float, line2[1:]))
            f.close()
            d = dict(zip(probes,range(len(probes))))
            selected_inds = [d[x] for x in input_probes]
            params = (kw, selected_inds, array(rows))
        return method, probes, input_probes, params
    except:
        print("Error reading imputation model from %s."%filename)
        return

def save_probe_list(probes, filename):
    try:
        f = open(filename, "wb")
        for probe in probes:
            f.write(probe+os.linesep)
        f.close()
        return True
    except:
        print("Error saving probe list to %s."%filename)
        return False

def save_imputation_model(method, m, num_probes, all_probes, filename):
    if method in ['RGE','SR']:
        selected_probes = m.get_selected_genes(num_probes)
        params = m.get_imputation_params(num_probes)
        try:
            f = open(filename, "wb")
            f.write(method+os.linesep)
            f.write('\t'.join(['target probe']+[all_probes[i] for i in selected_probes]+['bias'])+os.linesep)
            for i in xrange(params.shape[0]):
                f.write('\t'.join([all_probes[i]]+['%s'%x for x in params[i]])+os.linesep)
            f.close()
            return True
        except:
            print("Error saving imputation model to %s."%filename)
            return False
    elif method in ['LWA']:
        selected_probes = m.get_selected_genes(num_probes)
        params = m.get_imputation_params(num_probes)
        try:
            f = open(filename, "wb")
            f.write(method+os.linesep)
            f.write('\t'.join([all_probes[i] for i in selected_probes])+os.linesep)
            f.write('%f'%params+os.linesep)
            for i in xrange(m.data.shape[0]):
                f.write('\t'.join([all_probes[i]]+['%s'%x for x in m.data[i]])+os.linesep)
            f.close()
            return True
        except:
            raise
            print("Error saving imputation model to %s."%filename)
            return False
    else:
        print("Error saving imputation model: only RGE, SR and LWA models can be saved.")
        return False

def psi_initialize(method):
    if method == 'RGE':
        sigmasqrs = 1.25**arange(-40.0,20.0)
        m = rge.L2CovEstSelectionImputation(sigmasqrs, None)
    elif method == 'SR':
        m = sr.L1InfPSI()
    elif method == 'LWA':
        m = lwa.LWAPSI()
    return m

def psi_select(expr, probes, num_probes, m, force_include = []):
    m.set_probes_to_include(force_include)
    m.train(expr, num_probes)
    selected_probe_indices = m.get_selected_genes(num_probes)
    selected_probes = [probes[x] for x in selected_probe_indices]
    return selected_probes, m

if __name__ == "__main__":
    main()
