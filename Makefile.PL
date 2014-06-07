#!/usr/bin/perl
use strict;
use warnings;
use ExtUtils::MakeMaker;

my %params = (
	'NAME'          => 'AI::FANN::Evolving',
	'ABSTRACT_FROM' => 'lib/AI/FANN/Evolving.pm',
	'VERSION_FROM'  => 'lib/AI/FANN/Evolving.pm',
	'EXE_FILES'     => [ 'script/aivolver' ],
	'AUTHOR'        => 'Rutger Vos <rutger.vos@naturalis.nl>',
	'PREREQ_PM'     => { 	
		'AI::FANN' => 0,
		'Algorithm::Genetic::Diploid' => 0,	
	},
);

if ( $ExtUtils::MakeMaker::VERSION ge '6.46' ) {
    $params{'META_MERGE'} = {
        'resources' => {
            'bugtracker'  => 'https://github.com/naturalis/ai-fann-evolving/issues',
            'license'     => 'mit',
            'repository'  => {
            	'type' => 'git',
            	'url'  => 'git://github.com/naturalis/ai-fann-evolving.git',
            	'web'  => 'http://github.com/naturalis/ai-fann-evolving',
            }
        },
    };
}


WriteMakefile( %params );
