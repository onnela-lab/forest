''' Some basic code for testing whether forest modules can be imported.

'''


def test_imports(subpackage_name, module_names):
    n_success = 0
    n_failure = 0
    print('Testing imports from %s...' % subpackage_name)
    for m in module_names:
        test_import = 'from %s.%s import *' % (subpackage_name, m)
        try:
            exec(test_import)
            n_success += 1
        except Exception:
            print('    Failed to import %s.%s' % (subpackage_name, m))
            n_failure += 1
    print('Successful imports: %s' % n_success)
    print('Failed imports: %s \n' % n_failure)


# Test imports for forest.willow
test_imports('forest.willow',
             ['log_stats'])

# Test imports for forest.jasmine
test_imports('forest.jasmine',
             ['data2mobmat', 'mobmat2traj', 'sogp_gps', 'traj2stats',
              'simulate_gps_data'])

# Test imports for forest.poplar
test_imports('forest.poplar.classes',
             ['history', 'registry', 'template', 'trackers'])
test_imports('forest.poplar.constants',
             ['misc', 'time'])
test_imports('forest.poplar.functions',
             ['helpers', 'holidays', 'io', 'log', 'time', 'timezone'])
test_imports('forest.poplar.legacy',
             ['common_funcs'])
test_imports('forest.poplar.raw',
             ['doc', 'readers'])
